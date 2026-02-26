"""SD3-specific training utilities for annealing guidance."""
import os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import LaionDataset


def get_data_loader(config):
    batch_size = config["training"]["batch_size"]
    image_root = os.environ.get("ANNEALING_GUIDANCE_IMAGE_ROOT") or config["training"]["image_root"]
    image_root = os.path.expandvars(os.path.expanduser(str(image_root)))

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Training dataset folder does not exist. image_root={image_root!r}. "
            "Set ANNEALING_GUIDANCE_IMAGE_ROOT=/path/to/images"
        )

    print(f"Using training image_root: {image_root}", flush=True)
    dataset = LaionDataset(image_root)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.manual_seed(config['seed']),
        pin_memory=True
    )
    return dataloader


def save_model(config, guidance_scale_model, step, timestamp, final=False):
    dict_to_save = {
        'config': config,
        'model_state_dict': guidance_scale_model.state_dict(),
        'model_config': config.get('guidance_scale_model', {}),
        'step': step,
    }
    out_dir = config['training']['out_dir']
    checkpoint_dir = f'{out_dir}/checkpoints_sd3_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    if final:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_final.pt'
    else:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_step_{step}.pt'

    torch.save(dict_to_save, checkpoint_path)
    return os.path.abspath(checkpoint_path)


def get_timestep(pipeline, batch_size=1):
    """Sample random timesteps for SD3 flow matching."""
    n_timesteps = len(pipeline.scheduler.timesteps)
    timesteps_indices = torch.randint(1, n_timesteps, size=[batch_size])
    timesteps = pipeline.scheduler.timesteps[n_timesteps - 1 - timesteps_indices]
    timesteps = timesteps.to(device=pipeline.device)
    return timesteps


def calc_loss(noise_pred, noise_gt, delta_t_minus_one, l):
    # Compute loss in float32 to avoid overflow
    noise_pred = noise_pred.float()
    noise_gt = noise_gt.float()
    delta_t_minus_one = delta_t_minus_one.float()
    l = l.float()

    loss = torch.tensor(0.0, device=noise_pred.device)

    # epsilon loss
    squared_errors = ((noise_pred - noise_gt) ** 2).mean(dim=[1, 2, 3])
    eps_loss = ((1 - l) * squared_errors).mean()
    loss += eps_loss

    # delta loss
    squared_errors = (delta_t_minus_one ** 2).mean(dim=[1, 2, 3])
    diff_loss = (l * squared_errors).mean()
    loss += diff_loss

    return loss


def encode_prompt_sd3(pipeline, prompt):
    """Encode prompts for SD3 (three text encoders)."""
    device = pipeline.device
    dtype = pipeline.transformer.dtype

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    # Concatenate for CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds


def denoise_single_step_sd3(pipeline, latents, prompt_embeds, pooled_prompt_embeds, timestep):
    """Single denoising step for SD3 transformer."""
    timesteps = torch.cat([timestep] * 2)  # duplicate for uncond pred
    latent_model_input = torch.cat([latents] * 2)

    noise_pred = pipeline.transformer(
        hidden_states=latent_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]

    return noise_pred


def to_noisy_latents_sd3(pipeline, image, timestep, size=(1024, 1024)):
    """Encode image to latent and add noise for SD3 (flow matching).

    Returns (noisy_latents, velocity_gt) where velocity_gt = noise - clean_latents.
    SD3 uses flow matching: z_t = (1-sigma)*x_0 + sigma*epsilon, and the model
    predicts velocity v = epsilon - x_0.  The correct training target for the
    guidance scale MLP is therefore this velocity, NOT the raw noise.
    """
    with torch.no_grad():
        vae = pipeline.vae.to(torch.float32)
        image = image.to(device=vae.device, dtype=vae.dtype)
        image = torch.nn.functional.interpolate(image, size=size, mode='bilinear')
        latents = vae.encode(image).latent_dist.sample(generator=None)
        # SD3 uses different scaling
        latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor

    target_dtype = pipeline.transformer.dtype
    latents = latents.to(dtype=target_dtype)
    noise = torch.randn_like(latents)

    # SD3 uses flow matching - compute sigma from timestep
    # For flow matching: noisy = (1 - sigma) * x + sigma * noise
    # where sigma = timestep / 1000 (timesteps are in [0, 1000] range)
    sigma = (timestep.float() / 1000.0).to(dtype=target_dtype, device=latents.device)
    sigma = sigma.view(-1, 1, 1, 1)  # broadcast for batch
    noisy_latents = (1 - sigma) * latents + sigma * noise

    # Flow matching velocity target: v = epsilon - x_0
    velocity_gt = noise - latents

    return noisy_latents, velocity_gt


def linear_schedule(t: torch.Tensor, tau1: float, tau2: float) -> torch.Tensor:
    gamma = (tau2 - t) / (tau2 - tau1)
    gamma = torch.clamp(gamma, min=0.0, max=1.0)
    return gamma


def add_noise_to_prompt(y, gamma, noise_scale, psi, rescale=False):
    eps = 1e-6
    y_dtype = y.dtype
    y_f = y.float()
    gamma_f = gamma.to(device=y.device, dtype=torch.float32).view(-1, *[1] * (y.ndim - 1))
    noise_f = torch.randn_like(y_f)

    y_noised_f = torch.sqrt(gamma_f) * y_f + noise_scale * torch.sqrt(1 - gamma_f) * noise_f

    if not rescale:
        return y_noised_f.to(dtype=y_dtype)

    dims = tuple(range(1, y.ndim)) if y.ndim > 1 else ()
    y_mean, y_std = y_f.mean(dims, keepdim=True), y_f.std(dims, keepdim=True) + eps
    yn_mean, yn_std = y_noised_f.mean(dims, keepdim=True), y_noised_f.std(dims, keepdim=True) + eps

    y_scaled = (y_noised_f - yn_mean) / yn_std * y_std + y_mean
    out = psi * y_scaled + (1 - psi) * y_noised_f
    return out.to(dtype=y_dtype)


def prompt_add_noise_sd3(
    prompt_embeds,
    pooled_prompt_embeds,
    timestep,
    n_timesteps,
    add_noise,
    noise_scale,
    rescale,
    psi,
    t1,
    t2
):
    if add_noise:
        t = timestep / n_timesteps
        gamma = linear_schedule(t, t1, t2)

        # Split to negative and positive embeds
        negative_prompt_embeds, cond_prompt_embeds = prompt_embeds.chunk(2)
        cond_prompt_embeds = add_noise_to_prompt(cond_prompt_embeds, gamma, noise_scale, psi, rescale=rescale)
        prompt_embeds = torch.cat([negative_prompt_embeds, cond_prompt_embeds], dim=0)

        negative_pooled, pooled_cond = pooled_prompt_embeds.chunk(2)
        pooled_cond = add_noise_to_prompt(pooled_cond, gamma, noise_scale, psi, rescale=rescale)
        pooled_prompt_embeds = torch.cat([negative_pooled, pooled_cond], dim=0)

    return prompt_embeds, pooled_prompt_embeds
