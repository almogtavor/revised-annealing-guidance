"""SD3-specific utilities for annealing guidance.

Contains only model-specific helpers (load, encode, denoise, noisy latents, CADS).
Forward pass lives in scripts/train.py; shared functions in train_utils.py.
"""
import os
import glob
import subprocess
import torch
from src.utils.train_utils import add_noise_to_prompt, linear_schedule, get_timestep


def load_models(config, device):
    """Load SD3 pipeline + guidance MLP for training."""
    from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
    from src.model.guidance_scale_model import ScalarMLP

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    dtype = torch.float32

    pipeline = MyStableDiffusion3Pipeline.from_pretrained(
        config['diffusion']['model_id'], torch_dtype=dtype, token=hf_token)
    pipeline.to(device)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    for enc in [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]:
        if enc is not None:
            enc.requires_grad_(False)
    if hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
        pipeline.transformer.enable_gradient_checkpointing()

    model = ScalarMLP(**config['guidance_scale_model'])
    model.to(device, dtype=torch.float32)
    model.device, model.dtype = device, torch.float32
    return pipeline, model


def encode_prompt_sd3(pipeline, prompt):
    """Encode prompts using SD3's three text encoders."""
    device, dtype = pipeline.device, pipeline.transformer.dtype
    pe, npe, ppe, nppe = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    prompt_embeds = torch.cat([npe, pe], dim=0).to(device=device, dtype=dtype)
    pooled = torch.cat([nppe, ppe], dim=0).to(device=device, dtype=dtype)
    return prompt_embeds, pooled


def denoise_single_step_sd3(pipeline, latents, prompt_embeds, pooled, timestep):
    """Single SD3 transformer forward (uncond + cond)."""
    return pipeline.transformer(
        hidden_states=torch.cat([latents] * 2), timestep=torch.cat([timestep] * 2),
        encoder_hidden_states=prompt_embeds, pooled_projections=pooled,
        return_dict=False)[0]


def to_noisy_latents_sd3(pipeline, image, timestep, size=(1024, 1024)):
    """VAE encode + flow-matching noise. Returns (noisy_latents, velocity_gt)."""
    with torch.no_grad():
        vae = pipeline.vae.to(torch.float32)
        image = image.to(device=vae.device, dtype=vae.dtype)
        image = torch.nn.functional.interpolate(image, size=size, mode='bilinear')
        latents = vae.encode(image).latent_dist.sample(generator=None)
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    dt = pipeline.transformer.dtype
    latents = latents.to(dtype=dt)
    noise = torch.randn_like(latents)
    sigma = (timestep.float() / 1000.0).to(dtype=dt, device=latents.device).view(-1, 1, 1, 1)
    noisy_latents = (1 - sigma) * latents + sigma * noise
    return noisy_latents, noise - latents  # (z_t, v_gt = eps - x_0)


def prompt_add_noise_sd3(prompt_embeds, pooled, timestep, n_timesteps,
                         add_noise, noise_scale, rescale, psi, t1, t2):
    """CADS noise on SD3 prompt embeddings (conditional half only)."""
    if add_noise:
        gamma = linear_schedule(timestep / n_timesteps, t1, t2)
        neg, cond = prompt_embeds.chunk(2)
        prompt_embeds = torch.cat([neg, add_noise_to_prompt(cond, gamma, noise_scale, psi, rescale=rescale)])
        neg_p, cond_p = pooled.chunk(2)
        pooled = torch.cat([neg_p, add_noise_to_prompt(cond_p, gamma, noise_scale, psi, rescale=rescale)])
    return prompt_embeds, pooled



def run_auto_sample(config):
    """Find latest checkpoint and submit sampling SLURM job."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = config['training']['out_dir']
    pts = sorted(glob.glob(os.path.join(out_dir, 'checkpoints_*', 'checkpoint_step_*.pt')),
                 key=os.path.getmtime)
    if not pts:
        print("No checkpoints found for auto-sampling.", flush=True)
        return
    latest = os.path.abspath(pts[-1])
    lr = config.get('training', {}).get('optimizer_kwargs', {}).get('lr')
    label = config.get('training', {}).get('label')
    if lr is not None:
        ckpt_id = f"sd3_{lr}_{label}" if label else f"sd3_lr{lr}"
    else:
        ckpt_id = os.path.basename(os.path.dirname(latest))
    script = os.path.join(repo, "submit_sd3_sample.sh")
    os.makedirs(os.path.join(repo, "logs", "sampling"), exist_ok=True)
    print(f"\n{'='*60}\nSUBMITTING SAMPLING JOB: {ckpt_id}\n{'='*60}\n", flush=True)
    export_vars = f"ALL,SD3_SAMPLE_CHECKPOINT={latest},SD3_SAMPLE_CHECKPOINT_ID={ckpt_id}"
    result = subprocess.run(
        ["sbatch", "--export", export_vars, script],
        cwd=repo, capture_output=True, text=True)
    print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}", flush=True)

    # Also submit fig2 comparison job
    fig2_script = os.path.join(repo, "submit_sampling_woman_black_dress.sh")
    if os.path.exists(fig2_script):
        print(f"\n{'='*60}\nSUBMITTING FIG2 COMPARISON JOB\n{'='*60}\n", flush=True)
        fig2_vars = f"ALL,FIG2_CHECKPOINT={latest}"
        result2 = subprocess.run(
            ["sbatch", "--export", fig2_vars, fig2_script],
            cwd=repo, capture_output=True, text=True)
        print(result2.stdout.strip(), flush=True)
        if result2.returncode != 0:
            print(f"fig2 sbatch failed: {result2.stderr.strip()}", flush=True)
