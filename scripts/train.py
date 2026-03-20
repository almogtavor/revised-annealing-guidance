import math
import os
import sys
import time
import torch
import datetime
import tqdm

# Allow running as `python scripts/train.py` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.utils.model_utils as model_utils
import src.utils.train_utils as train_utils
import src.utils.train_utils_sd3 as train_utils_sd3
import src.utils.wandb_utils as wb
import src.utils.resume_utils as resume_utils
import src.utils.ddp_utils as ddp_utils; ddp_utils.setup()


def train(config, pipeline, model, optimizer, dataloader, forward_fn=None, resume_step=0):
    if forward_fn is None:
        forward_fn = forward_pass
    train_config = config['training']
    max_steps = train_config['max_steps']
    max_epochs = math.ceil(max_steps / len(dataloader))
    accumulation_steps = max(train_config.get('accumulation_steps', 1), 1)
    grad_clip = train_config.get('grad_clip', 1.0)

    train_end = False
    global_step = 0
    nan_count = 0

    datetime_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(max_epochs):
        epoch_start = time.time()
        for batch in tqdm.tqdm(dataloader, miniters=100, mininterval=60):
            model.train()
            prompts, images = batch

            images = images.to(pipeline.device)

            if global_step < resume_step:
                global_step += 1
                continue

            result = forward_fn(config, pipeline, model, images, prompts)
            if isinstance(result, dict):
                loss = result['loss']
                extra_metrics = {k: v for k, v in result.items() if k != 'loss'}
            else:
                loss = result
                extra_metrics = None

            # Skip NaN losses to prevent weight corruption
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count % 10 == 1:
                    print(f"WARNING: NaN/Inf loss at step {global_step} (total skipped: {nan_count})", flush=True)
                optimizer.zero_grad()
                global_step += 1
                continue

            loss = loss / accumulation_steps  # Normalize loss by accumulation steps
            loss.backward()
            wb.log_train(global_step, loss.item() * accumulation_steps, model, extra_metrics=extra_metrics)

            if (global_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if global_step > 0 and global_step % config['training']['save_interval'] == 0:
                print(f"Saving model at step {global_step}...")
                resume_utils.save_checkpoint(config, model, optimizer, global_step, datetime_timestamp)


            global_step += 1
            if global_step > max_steps:
                train_end = True
                break

        if train_end:
            break

        # Single marker line between epochs for SLURM logs
        epoch_seconds = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{max_epochs} finished in {epoch_seconds:.1f}s (global_step={global_step})",
            flush=True,
        )


def forward_pass(
    config,
    pipeline,
    model,
    images,
    prompts,
):
    batch_size = images.size(0)
    
    # Select lambda values
    l = torch.rand(batch_size).to(pipeline.unet.device)

    # Select timestep values
    timestep = train_utils.get_timestep(pipeline, batch_size=batch_size)

    # Get noisy latents and ground truth noise
    # x_0 -> z_t
    noisy_latents, noise_gt = train_utils.to_noisy_latents(pipeline, images, timestep) # (z_t, eps)

    # Get prompt embeddings
    with torch.no_grad():
        prompt_embeds, added_cond_kwargs = train_utils.encode_prompt(pipeline, prompts)

    # Use CADS to add noise to conditioning signal (if enabled)
    prompt_embeds, added_cond_kwargs = train_utils.prompt_add_noise(
        prompt_embeds,
        added_cond_kwargs,
        timestep,
        pipeline.scheduler.config['num_train_timesteps'],
        **config['training']['prompt_noise']
    )

    # Predict epsilon_null + epsilon_cond
    noise_pred = train_utils.denoise_single_step(
        pipeline,
        noisy_latents,
        prompt_embeds,
        timestep,
        added_cond_kwargs,
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
    # Predict guidance scale
    # noise_pred, guidance_scale, _ = pipeline.perform_guidance(noise_pred_uncond, noise_pred_text, model.cfg, timestep, model, l=l)
    guidance_scale_pred = model(timestep, l, noise_pred_uncond, noise_pred_text)
    
    # Apply classifier free guidance
    noise_pred = noise_pred_uncond + guidance_scale_pred * (noise_pred_text - noise_pred_uncond)
    
    # Renoise to next latent
    # z_t -> z_{t+1}
    results =  pipeline.scheduler.step(noise_pred, timestep, noisy_latents, return_dict=True, noise_pred_uncond=noise_pred_uncond)
    pred_latents_prev = results['prev_sample']


    # Calculate delta_t_minus_one
    # Reverse scheduler_timesteps for ascending order in search
    scheduler_timesteps = pipeline.scheduler.timesteps.clone().to(device=timestep.device)
    scheduler_timesteps_reversed = scheduler_timesteps.flip(0)

    # Find indices in the reversed array
    timestep_indices = torch.searchsorted(scheduler_timesteps_reversed, timestep, right=True) - 1
    valid_indices = (timestep_indices - 1) >= 0
    if valid_indices.any():
        timestep_indices = timestep_indices[valid_indices]
        timestep_prev = scheduler_timesteps_reversed[timestep_indices - 1]
        pred_latents_prev = pred_latents_prev[valid_indices]
        prompt = [p for p, valid in zip(prompts, valid_indices.cpu().numpy()) if valid]

        with torch.no_grad():
            prompt_embeds, added_cond_kwargs = train_utils.encode_prompt(pipeline, prompt)

        noise_pred_prev = train_utils.denoise_single_step(
            pipeline,
            pred_latents_prev,
            prompt_embeds,
            timestep_prev,
            {k: v[valid_indices] if isinstance(v, torch.Tensor) and v.shape[0] == valid_indices.shape[0] else v
            for k, v in added_cond_kwargs.items()},
        )[0]

        noise_pred_uncond_prev, noise_pred_text_prev = noise_pred_prev.chunk(2, dim=0)
        delta_t_minus_one = noise_pred_uncond_prev - noise_pred_text_prev

    # calc loss
    ema_normalize = config['training'].get('ema_loss_normalization', True)
    loss = train_utils.calc_loss(noise_pred, noise_gt, delta_t_minus_one, l, ema_normalize=ema_normalize)

    return loss


def forward_pass_sd3(
    config,
    pipeline,
    model,
    images,
    prompts,
):
    """SD3 forward pass: flow-matching adaptation of Algorithm 1."""
    B = images.size(0)
    dtype = pipeline.transformer.dtype

    l = torch.rand(B).to(pipeline.device)
    fixed_lam = config['training'].get('fixed_lambda')
    if fixed_lam is not None:
        l = torch.full_like(l, fixed_lam)
    timestep = train_utils.get_timestep(pipeline, batch_size=B)
    noisy_latents, velocity_gt = train_utils_sd3.to_noisy_latents_sd3(pipeline, images, timestep)

    with torch.no_grad():
        pe, ppe = train_utils_sd3.encode_prompt_sd3(pipeline, prompts)
    pe, ppe = train_utils_sd3.prompt_add_noise_sd3(
        pe, ppe, timestep, pipeline.scheduler.config.get('num_train_timesteps', 1000),
        **config['training']['prompt_noise'])

    # Pass 1: SD3 at z_t (frozen)
    with torch.no_grad():
        pred = train_utils_sd3.denoise_single_step_sd3(pipeline, noisy_latents, pe, ppe, timestep)
    vu, vt = pred.float().chunk(2)
    del pred

    w = model(timestep.float(), l, vu, vt)
    v_guided = vu + w * (vt - vu)

    # CFG++ step (flow matching)
    n_steps = config['diffusion'].get('num_timesteps', 50)
    st = (timestep.float() / 1000.0).to(device=noisy_latents.device)
    st1 = (st - 1.0 / n_steps).clamp(min=1e-4)
    st_, st1_ = st.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1)
    zf = noisy_latents.float()
    x0 = zf - st_ * v_guided
    eps_u = zf + (1.0 - st_) * vu
    z_next = (1.0 - st1_) * x0 + st1_ * eps_u

    # Pass 2: direct delta loss (full backprop through frozen transformer)
    t_next = (st1 * 1000.0).to(dtype=timestep.dtype, device=timestep.device)
    del x0, eps_u, zf
    torch.cuda.empty_cache()

    pred2 = train_utils_sd3.denoise_single_step_sd3(pipeline, z_next.to(dtype=dtype), pe, ppe, t_next)
    vu2, vt2 = pred2.float().chunk(2)
    delta = vt2 - vu2

    # Same loss structure as SDXL but with velocity instead of noise
    ema_normalize = config['training'].get('ema_loss_normalization', True)
    loss = train_utils.calc_loss(v_guided, velocity_gt.float(), delta, l, ema_normalize=ema_normalize)

    eps_val = ((1 - l) * ((v_guided - velocity_gt.float()) ** 2).mean(dim=[1, 2, 3])).mean()
    diff_val = (l * (delta ** 2).mean(dim=[1, 2, 3])).mean()

    # Delta/velocity diagnostics
    delta_t = (vt - vu)  # delta at current timestep
    delta_norm = delta_t.view(B, -1).norm(dim=1).mean().item()
    delta_next_norm = delta.view(B, -1).norm(dim=1).mean().item()
    return {
        'loss': loss,
        'train/eps_loss': eps_val.item(),
        'train/diff_loss': diff_val.item(),
        'train/delta_norm': delta_norm,
        'train/delta_next_norm': delta_next_norm,
    }


if not torch.cuda.is_available():
    print("FATAL: CUDA not available. Refusing to run on CPU.")
    sys.exit(1)
device = torch.device("cuda")

props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"GPU total VRAM (GiB): {props.total_memory / (1024**3):.2f}", flush=True)

config_path = os.environ.get('ANNEALING_GUIDANCE_CONFIG', 'scripts/config.yaml')
_, config = model_utils.load_config(config_path=config_path)
is_sd3 = 'stable-diffusion-3' in config['diffusion']['model_id']
if is_sd3:
    pipeline, guidance_scale_network = train_utils_sd3.load_models(config, device)
else:
    _, pipeline, guidance_scale_network = model_utils.load_models(config_path=config_path, device=device)
guidance_scale_network = ddp_utils.wrap(guidance_scale_network)

# Optional overrides (useful for SLURM sanity-check runs)
_env_max_steps = os.environ.get("ANNEALING_GUIDANCE_MAX_STEPS")
if _env_max_steps:
    config.setdefault("training", {})
    config["training"]["max_steps"] = int(_env_max_steps)

_env_save_interval = os.environ.get("ANNEALING_GUIDANCE_SAVE_INTERVAL")
if _env_save_interval:
    config.setdefault("training", {})
    config["training"]["save_interval"] = int(_env_save_interval)

print("Models/pipeline loaded; building optimizer and dataloader...", flush=True)

optimizer = torch.optim.AdamW(guidance_scale_network.parameters(), **config['training']['optimizer_kwargs'])
dataloader = train_utils.get_data_loader(config)

print(f"Dataloader ready: {len(dataloader)} batches/epoch", flush=True)

resume_step = resume_utils.maybe_resume(config, guidance_scale_network, optimizer)

forward_fn = forward_pass_sd3 if is_sd3 else forward_pass
wb.init_training(config, guidance_scale_network, n_samples=len(dataloader))
train(config, pipeline, guidance_scale_network, optimizer, dataloader, forward_fn=forward_fn, resume_step=resume_step)
wb.finish()
if is_sd3:
    train_utils_sd3.run_auto_sample(config)
