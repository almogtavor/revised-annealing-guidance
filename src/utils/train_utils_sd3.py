"""SD3-specific utilities for annealing guidance.

Contains only model-specific helpers (load, encode, denoise, noisy latents, CADS).
Forward pass lives in scripts/train.py; shared functions in train_utils.py.
"""
import os
import glob
import re
import subprocess
import torch
from src.utils.train_utils import add_noise_to_prompt, linear_schedule, get_timestep


def get_num_sampling_steps(config_or_diffusion, default=None):
    """Read num_sampling_steps from a config dict, with num_timesteps as backward-compat alias.

    Accepts either a full config or just the 'diffusion' subsection.
    """
    if config_or_diffusion is None:
        return default
    diff = config_or_diffusion.get('diffusion', config_or_diffusion)
    if diff is None:
        return default
    return diff.get('num_sampling_steps') or diff.get('num_timesteps') or default


def get_prev_timestep(scheduler, timestep):
    """Return the previous discrete scheduler timestep for each sampled timestep."""
    scheduler_timesteps = scheduler.timesteps.to(device=timestep.device)
    scheduler_timesteps_reversed = scheduler_timesteps.flip(0)
    timestep_f = timestep.float()
    idx = torch.searchsorted(scheduler_timesteps_reversed, timestep_f, right=False)
    idx = idx.clamp(min=1)
    prev = scheduler_timesteps_reversed[idx - 1]
    return prev.to(dtype=timestep.dtype, device=timestep.device)


def load_models(config, device):
    """Load SD3 pipeline + guidance MLP for training."""
    from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
    from src.model.guidance_scale_model import ScalarMLP

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    fp16 = config.get('fp16', False)
    dtype = torch.float16 if fp16 else torch.float32

    cache_dir = config.get('training', {}).get('prompt_cache_dir')
    use_cache = cache_dir and os.path.isdir(cache_dir)

    if use_cache:
        # Skip text encoders entirely — load only transformer + VAE
        # This allows training on smaller GPUs (e.g. 12GB TITAN Xp)
        print(f"Prompt cache found at {cache_dir}; loading without text encoders.", flush=True)
        pipeline = MyStableDiffusion3Pipeline.from_pretrained(
            config['diffusion']['model_id'], torch_dtype=dtype, token=hf_token,
            text_encoder=None, text_encoder_2=None, text_encoder_3=None)
        pipeline.transformer.to(device)
        pipeline.vae.to(device)
    else:
        pipeline = MyStableDiffusion3Pipeline.from_pretrained(
            config['diffusion']['model_id'], torch_dtype=dtype, token=hf_token)
        pipeline.to(device)
        for enc in [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]:
            if enc is not None:
                enc.requires_grad_(False)

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    if hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
        pipeline.transformer.enable_gradient_checkpointing()

    mlp_kwargs = dict(config['guidance_scale_model'])
    mlp_kwargs.setdefault('num_timesteps', get_num_sampling_steps(config))
    model = ScalarMLP(**mlp_kwargs)
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


_neg_cache = {}  # module-level cache for negative embeddings


def load_cached_prompt_sd3(cache_dir, image_root, image_paths, device):
    """Load pre-cached prompt embeddings from disk.

    Returns (prompt_embeds, pooled) in the same format as encode_prompt_sd3:
      prompt_embeds: [neg; pos] concatenated along batch dim
      pooled: [neg_pooled; pos_pooled] concatenated along batch dim
    """
    dtype = torch.float32
    B = len(image_paths)

    # Load shared negative embeddings (cached once)
    if cache_dir not in _neg_cache:
        neg = torch.load(os.path.join(cache_dir, "_negative.pt"), map_location="cpu")
        _neg_cache[cache_dir] = neg
    neg = _neg_cache[cache_dir]
    npe = neg["negative_prompt_embeds"].unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
    nppe = neg["negative_pooled_prompt_embeds"].unsqueeze(0).expand(B, -1).to(device=device, dtype=dtype)

    # Load per-prompt positive embeddings
    pe_list, ppe_list = [], []
    for img_path in image_paths:
        rel = os.path.relpath(img_path, image_root)
        cache_path = os.path.join(cache_dir, rel.replace(".jpg", ".pt"))
        cached = torch.load(cache_path, map_location="cpu")
        pe_list.append(cached["prompt_embeds"])
        ppe_list.append(cached["pooled_prompt_embeds"])
    pe = torch.stack(pe_list).to(device=device, dtype=dtype)
    ppe = torch.stack(ppe_list).to(device=device, dtype=dtype)

    prompt_embeds = torch.cat([npe, pe], dim=0)
    pooled = torch.cat([nppe, ppe], dim=0)
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
    latest_dir = os.path.dirname(latest)
    target_images = config['training'].get('max_images')
    if target_images is None:
        target_images = config['training'].get('max_steps')
    if target_images is not None:
        numbered = []
        for pt in pts:
            if os.path.dirname(pt) != latest_dir:
                continue
            m = re.search(r'checkpoint_step_(\d+)\.pt$', pt)
            if m:
                numbered.append((int(m.group(1)), pt))
        if numbered:
            numbered.sort()
            ge = [pt for n, pt in numbered if n >= target_images]
            latest = os.path.abspath(ge[0] if ge else numbered[-1][1])
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
    n_steps = get_num_sampling_steps(config, default='')
    job_name = f"sample-{ckpt_id}" if not n_steps else f"sample-steps{n_steps}"
    result = subprocess.run(
        ["sbatch", "--job-name", job_name, "--export", export_vars, script],
        cwd=repo, capture_output=True, text=True)
    print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}", flush=True)

    # Also submit fig2 comparison job
    fig2_script = os.path.join(repo, "submit_sd3_sampling_woman_black_dress.sh")
    if os.path.exists(fig2_script):
        print(f"\n{'='*60}\nSUBMITTING FIG2 COMPARISON JOB\n{'='*60}\n", flush=True)
        fig2_vars = f"ALL,FIG2_CHECKPOINT={latest}"
        result2 = subprocess.run(
            ["sbatch", "--export", fig2_vars, fig2_script],
            cwd=repo, capture_output=True, text=True)
        print(result2.stdout.strip(), flush=True)
        if result2.returncode != 0:
            print(f"fig2 sbatch failed: {result2.stderr.strip()}", flush=True)
