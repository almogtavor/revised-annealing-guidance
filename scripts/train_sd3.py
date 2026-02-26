"""SD3 annealing guidance training script."""
import math
import os
import sys
import time
import torch
import datetime
import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
from src.model.guidance_scale_model import ScalarMLP
import src.utils.train_utils_sd3 as train_utils
import src.utils.wandb_utils as wb


def load_sd3_pipeline(config, device, dtype):
    """Load SD3 pipeline for training."""
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    pipeline = MyStableDiffusion3Pipeline.from_pretrained(
        config['diffusion']['model_id'],
        torch_dtype=dtype,
        token=hf_token,
    )
    pipeline.to(device)

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    # Freeze model weights
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)

    # Per-block gradient checkpointing: saves ~12x activation memory during
    # the 2nd forward pass (only block inputs are stored, activations recomputed
    # during backward).  No-op under torch.no_grad() so pass 1 is unaffected.
    if hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
        pipeline.transformer.enable_gradient_checkpointing()
    if pipeline.text_encoder is not None:
        pipeline.text_encoder.requires_grad_(False)
    if pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.requires_grad_(False)
    if pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.requires_grad_(False)

    return pipeline


def load_guidance_model(config, device, dtype):
    """Load guidance scale MLP.

    Always use float32 for the trainable MLP to avoid NaN from float16 gradient overflow.
    The SD3 pipeline can stay in float16 for memory savings.
    """
    model = ScalarMLP(**config['guidance_scale_model'])
    model.to(device, dtype=torch.float32)
    model.device = device
    model.dtype = torch.float32
    return model


def format_time(seconds):
    """Format seconds into human readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def print_training_summary(
    config,
    total_time,
    completed_epochs,
    max_epochs,
    global_step,
    max_steps,
    num_samples,
    batch_size,
    peak_gpu_memory_gb,
    final_checkpoint_path,
    last_checkpoint_path,
):
    """Print a comprehensive training summary."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    # GPU Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        print(f"GPU Type:              {gpu_name}")
        print(f"GPU Total VRAM:        {total_vram_gb:.2f} GiB")
        print(f"Peak GPU Memory Used:  {peak_gpu_memory_gb:.2f} GiB ({100*peak_gpu_memory_gb/total_vram_gb:.1f}%)")
    else:
        print("GPU Type:              CPU (no GPU)")

    print("-" * 60)

    # Training configuration
    print(f"Model ID:              {config['diffusion']['model_id']}")
    print(f"Batch Size:            {batch_size}")
    print(f"Learning Rate:         {config['training']['optimizer_kwargs'].get('lr', 'N/A')}")
    print(f"Accumulation Steps:    {config['training'].get('accumulation_steps', 1)}")

    print("-" * 60)

    # Dataset info
    print(f"Dataset Size:          {num_samples} samples")
    print(f"Batches per Epoch:     {math.ceil(num_samples / batch_size)}")

    print("-" * 60)

    # Training progress
    print(f"Completed Epochs:      {completed_epochs}/{max_epochs}")
    print(f"Completed Steps:       {global_step}/{max_steps}")
    print(f"Total Training Time:   {format_time(total_time)}")
    if global_step > 0:
        print(f"Avg Time per Step:     {total_time/global_step:.3f}s")
        print(f"Avg Time per Epoch:    {format_time(total_time/completed_epochs)}")

    print("-" * 60)

    # Checkpoint info
    print(f"Save Interval:         every {config['training']['save_interval']} steps")
    if last_checkpoint_path:
        print(f"Last Checkpoint:       {last_checkpoint_path}")
    print(f"Final Checkpoint:      {final_checkpoint_path}")

    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60 + "\n")


def train(config, pipeline, model, optimizer, dataloader):
    train_config = config['training']
    max_steps = train_config['max_steps']
    max_epochs = math.ceil(max_steps / len(dataloader))
    accumulation_steps = max(train_config.get('accumulation_steps', 1), 1)

    train_end = False
    global_step = 0
    datetime_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Track training metrics
    training_start_time = time.time()
    peak_gpu_memory_gb = 0.0
    last_checkpoint_path = None
    completed_epochs = 0

    for epoch in range(max_epochs):
        epoch_start = time.time()
        for batch in tqdm.tqdm(dataloader, miniters=100, mininterval=60):
            model.train()
            prompts, images = batch
            images = images.to(pipeline.device)

            loss = forward_pass(config, pipeline, model, images, prompts)

            # Skip NaN losses to prevent weight corruption
            if torch.isnan(loss).any():
                optimizer.zero_grad()
                global_step += 1
                continue

            loss = loss / accumulation_steps
            loss.backward()
            wb.log_train(global_step, loss.item() * accumulation_steps, model)

            if (global_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

                # NaN detection: stop early if weights diverged
                if global_step % 200 == 0:
                    first_param = next(model.parameters())
                    if torch.isnan(first_param).any():
                        print(f"\nERROR: NaN detected in model weights at step {global_step}! Stopping.", flush=True)
                        train_end = True
                        break

            # Track peak GPU memory
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / (1024**3)
                peak_gpu_memory_gb = max(peak_gpu_memory_gb, current_memory)

            if global_step > 0 and global_step % config['training']['save_interval'] == 0:
                print(f"Saving model at step {global_step}...")
                last_checkpoint_path = train_utils.save_model(config, model, global_step, datetime_timestamp)

            global_step += 1
            if global_step > max_steps:
                train_end = True
                break

        completed_epochs = epoch + 1
        if train_end:
            break

        epoch_seconds = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{max_epochs} finished in {epoch_seconds:.1f}s (global_step={global_step})", flush=True)

    # Save final checkpoint
    final_checkpoint_path = train_utils.save_model(config, model, global_step, datetime_timestamp, final=True)

    # Print training summary
    total_training_time = time.time() - training_start_time
    print_training_summary(
        config=config,
        total_time=total_training_time,
        completed_epochs=completed_epochs,
        max_epochs=max_epochs,
        global_step=global_step,
        max_steps=max_steps,
        num_samples=len(dataloader.dataset),
        batch_size=config['training']['batch_size'],
        peak_gpu_memory_gb=peak_gpu_memory_gb,
        final_checkpoint_path=final_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )

    return final_checkpoint_path


def forward_pass(config, pipeline, model, images, prompts):
    """Two-pass training as in the paper (Eq. 6-8).

    Pass 1 (no_grad): SD3 at z_t  → v_uncond_t, v_text_t
    MLP: predict guidance scale w from (t, λ, v_uncond_t, v_text_t)
    CFG++ step: z_{t-1} = (z_t - σ_t · v_guided) + σ_{t-1} · v_uncond_t
    Pass 2 (with input grad): SD3 at z_{t-1} → v_uncond_{t-1}, v_text_{t-1}

    Loss (Eq. 6):  L = λ · L_δ  +  (1-λ) · L_ε
      L_ε (Eq. 8): ||v_guided - v_gt||²          (ε-loss at z_t)
      L_δ (Eq. 7): ||v_text(z_{t-1}) - v_uncond(z_{t-1})||²  (δ-loss at z_{t-1})

    Gradient path for L_δ:
      loss → δ_{t-1} → SD3(z_{t-1}) [input jacobian] → z_{t-1} → w_θ
    SD3 weights stay frozen; only input grads are computed in pass 2.
    """
    batch_size = images.size(0)
    dtype = pipeline.transformer.dtype  # float16

    # Random λ and timestep per sample
    l = torch.rand(batch_size).to(pipeline.device)
    timestep = train_utils.get_timestep(pipeline, batch_size=batch_size)

    # Noisy latents and flow-matching velocity target  v_gt = ε - x_0
    noisy_latents, velocity_gt = train_utils.to_noisy_latents_sd3(pipeline, images, timestep)

    # Encode prompts (frozen text encoders, no grad)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = train_utils.encode_prompt_sd3(pipeline, prompts)

    # CADS prompt noise (if enabled)
    prompt_embeds, pooled_prompt_embeds = train_utils.prompt_add_noise_sd3(
        prompt_embeds,
        pooled_prompt_embeds,
        timestep,
        pipeline.scheduler.config.get('num_train_timesteps', 1000),
        **config['training']['prompt_noise']
    )

    # === PASS 1: SD3 forward at z_t (frozen weights, no activation storage) ===
    with torch.no_grad():
        noise_pred = train_utils.denoise_single_step_sd3(
            pipeline, noisy_latents, prompt_embeds, pooled_prompt_embeds, timestep,
        )
    v_uncond_t, v_text_t = noise_pred.chunk(2)
    v_uncond_f32 = v_uncond_t.float()
    v_text_f32 = v_text_t.float()
    del noise_pred, v_uncond_t, v_text_t  # free float16 copies

    # MLP predicts guidance scale w  (float32, trainable)
    guidance_scale_pred = model(timestep.float(), l, v_uncond_f32, v_text_f32)

    # Guided velocity at z_t
    v_guided = v_uncond_f32 + guidance_scale_pred * (v_text_f32 - v_uncond_f32)

    # Sigma values: σ_t = timestep / 1000,  σ_{t-1} = σ_t − 1/n_steps
    n_timesteps = len(pipeline.scheduler.timesteps)
    sigma_t = (timestep.float() / 1000.0).to(device=noisy_latents.device)
    sigma_step = 1.0 / n_timesteps
    sigma_t1 = (sigma_t - sigma_step).clamp(min=1e-4)

    sigma_t_ = sigma_t.view(-1, 1, 1, 1)
    sigma_t1_ = sigma_t1.view(-1, 1, 1, 1)

    # CFG++ denoising step — flow-matching equivalent of DDPM Algorithm 1:
    #
    #   DDPM:  z_{t-1} = sqrt(ᾱ_{t-1}) · z_{0|t}  + sqrt(1-ᾱ_{t-1}) · ε^∅(z_t)
    #   Flow:  z_{t-1} = (1-σ_{t-1})   · x_0_pred  + σ_{t-1}          · ε_uncond
    #
    # where in flow matching:
    #   x_0_pred = z_t − σ_t · v_guided                (guided x_0 estimate)
    #   ε_uncond = z_t + (1−σ_t) · v_uncond            (uncond noise direction)
    #     derivation: z_t = (1-σ)·x_0 + σ·ε  →  ε = (z_t − (1-σ)·x_0) / σ
    #                 x_0_uncond = z_t − σ_t · v_uncond
    #                 ε_uncond = z_t + (1−σ_t) · v_uncond
    #
    # ∂z_{t-1}/∂w = −σ_t · (1−σ_{t-1}) · (v_text_t − v_uncond_t)  ← propagates grad
    noisy_latents_f32 = noisy_latents.float()
    x0_pred = noisy_latents_f32 - sigma_t_ * v_guided
    eps_uncond = noisy_latents_f32 + (1.0 - sigma_t_) * v_uncond_f32
    z_next = (1.0 - sigma_t1_) * x0_pred + sigma_t1_ * eps_uncond  # grad w.r.t. guidance_scale_pred

    # === PASS 2: SD3 forward at z_{t-1} — manual VJP to save VRAM ===
    #
    # Full autograd through SD3 (~2B params) would store the entire activation
    # graph simultaneously with the MLP graph → OOM on 24 GB GPUs.
    #
    # Instead we split the chain rule manually:
    #   ∂L_δ/∂w = ∂L_δ/∂z_{t-1} · ∂z_{t-1}/∂w
    #
    # Step A: forward SD3 on *detached* z_{t-1}, compute grad_z = ∂L_δ/∂z_{t-1}
    #         via torch.autograd.grad (graph is freed immediately after).
    # Step B: proxy loss = grad_z · z_next  (z_next still linked to MLP)
    #         → ∂proxy/∂w = grad_z · ∂z_next/∂w = ∂L_δ/∂w   ✓
    #
    # Peak VRAM = max(SD3_fwd+bwd, MLP_fwd+bwd) instead of SD3+MLP together.
    timestep_next = (sigma_t1 * 1000.0).to(dtype=timestep.dtype, device=timestep.device)

    # Free intermediate tensors from pass 1 / CFG++ step before pass 2
    del x0_pred, eps_uncond, noisy_latents_f32
    torch.cuda.empty_cache()

    # Detach z_next from MLP graph, cast to model dtype, enable input grad
    z_next_detached = z_next.detach().to(dtype=dtype).requires_grad_(True)

    # SD3 forward at z_{t-1} (weights frozen, only input grad tracked)
    noise_pred_next = train_utils.denoise_single_step_sd3(
        pipeline, z_next_detached, prompt_embeds, pooled_prompt_embeds, timestep_next,
    )
    v_uncond_next, v_text_next = noise_pred_next.chunk(2)

    # δ_{t-1} = v_text(z_{t-1}) − v_uncond(z_{t-1})  (Eq. 7)
    delta_next = v_text_next.float() - v_uncond_next.float()
    delta_loss_per_sample = (delta_next ** 2).mean(dim=[1, 2, 3])

    # Step A: VJP — compute ∂(λ·L_δ)/∂z_{t-1}, then free SD3 graph
    weighted_delta_sum = (l * delta_loss_per_sample).sum()
    grad_z = torch.autograd.grad(weighted_delta_sum, z_next_detached)[0]

    # Step B: proxy loss whose gradient w.r.t. MLP weights equals ∂L_δ/∂w
    delta_loss_proxy = (grad_z.float() * z_next).sum() / batch_size

    # ε-loss: ||v_guided − v_gt||² weighted by (1-λ)  (Eq. 8)
    eps_loss_per_sample = ((v_guided - velocity_gt.float()) ** 2).mean(dim=[1, 2, 3])
    eps_loss = ((1 - l) * eps_loss_per_sample).mean()

    # Total loss (Eq. 6): gradient is correct; value may differ from the analytical
    # L = mean_i[(1-λ_i)·L_ε_i + λ_i·L_δ_i] but that's fine for optimisation.
    loss = eps_loss + delta_loss_proxy

    return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU total VRAM (GiB): {props.total_memory / (1024**3):.2f}", flush=True)

    config_path = 'scripts/config_sd3.yaml'
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    # Env overrides
    _env_max_steps = os.environ.get("ANNEALING_GUIDANCE_MAX_STEPS")
    if _env_max_steps:
        config.setdefault("training", {})
        config["training"]["max_steps"] = int(_env_max_steps)

    _env_save_interval = os.environ.get("ANNEALING_GUIDANCE_SAVE_INTERVAL")
    if _env_save_interval:
        config.setdefault("training", {})
        config["training"]["save_interval"] = int(_env_save_interval)

    dtype = torch.float16 if config.get('low_memory', True) else torch.float32

    print("Loading SD3 pipeline...", flush=True)
    pipeline = load_sd3_pipeline(config, device, dtype)

    print("Loading guidance scale model...", flush=True)
    guidance_scale_network = load_guidance_model(config, device, dtype)

    print("Building optimizer and dataloader...", flush=True)
    optimizer = torch.optim.AdamW(guidance_scale_network.parameters(), **config['training']['optimizer_kwargs'])
    dataloader = train_utils.get_data_loader(config)

    print(f"Dataloader ready: {len(dataloader)} batches/epoch", flush=True)
    wb.init_training(config, guidance_scale_network)

    final_checkpoint_path = train(config, pipeline, guidance_scale_network, optimizer, dataloader)
    wb.finish()

    # Auto-sample after training: generate a small set of images to verify quality
    import subprocess
    sample_script = os.path.join(_REPO_ROOT, "scripts", "batch_sample_sd3.py")
    checkpoint_id = os.path.basename(os.path.dirname(final_checkpoint_path))
    sample_cmd = [
        sys.executable, "-u", sample_script,
        "--checkpoint", final_checkpoint_path,
        "--checkpoint_id", checkpoint_id,
        "--output_root", "results/images",
        "--figures", "fig_1",
        "--lambdas", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0",
        "--force",
    ]
    print(f"\n{'='*60}")
    print("AUTO-SAMPLING: generating fig_1 images with trained model...")
    print(f"{'='*60}\n", flush=True)
    subprocess.run(sample_cmd, cwd=_REPO_ROOT)
