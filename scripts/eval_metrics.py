"""Evaluate FID, CLIP score, and ImageReward on COCO 2017 val (5k images).

Replicates Tables 1-2 from the TIGER paper (arXiv:2506.24108).

Baselines (CFG, CFG++, APG) are generated once and cached in
  results/baseline_cache/
so subsequent runs (with different annealing checkpoints) skip them.

Usage:
  python scripts/eval_metrics.py \
      --checkpoint output/checkpoints_.../checkpoint_step_XXXXX.pt \
      --output_dir results/images/$SLURM_JOB_ID \
      [--label my_run_name] \
      [--skip_baselines]       # only generate annealing images
"""
import os
import sys
import json
import csv
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------
NUM_IMAGES = 1000
NUM_INFERENCE_STEPS = 28
SEED_OFFSET = 0  # seed = image_id (from COCO)

# Baselines: (dir_name, method, guidance_scale, use_cfgpp, use_apg)
BASELINE_CONFIGS = [
    ("cfg_w7.5",   "CFG",   7.5,  False, False),
    ("cfg_w10",    "CFG",   10.0, False, False),
    ("cfg_w12.5",  "CFG",   12.5, False, False),
    ("cfg_w15",    "CFG",   15.0, False, False),
    ("cfgpp_w0.6", "CFG++", 0.6,  True,  False),
    ("cfgpp_w0.8", "CFG++", 0.8,  True,  False),
    ("cfgpp_w1.0", "CFG++", 1.0,  True,  False),
    ("cfgpp_w1.2", "CFG++", 1.2,  True,  False),
    ("apg_w10",    "APG",   10.0, False, True),
    ("apg_w15",    "APG",   15.0, False, True),
    ("apg_w17.5",  "APG",   17.5, False, True),
    ("apg_w20",    "APG",   20.0, False, True),
]

# Annealing lambda values to evaluate
ANNEALING_LAMBDAS = [0.05, 0.4, 0.7, 0.8]
ANNEALING_GUIDANCE_SCALE = 7.0

_BASELINE_CACHE_ROOT = os.path.join(_REPO_ROOT, "results", "baseline_cache")

# ---------------------------------------------------------------------------
# COCO 2017 val helpers
# ---------------------------------------------------------------------------
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"


def get_coco_prompts(coco_dir: str):
    """Load first caption per image from COCO 2017 val, return list of (image_id, caption)."""
    import json as _json
    ann_file = os.path.join(coco_dir, "annotations", "captions_val2017.json")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_file}. "
            f"Download from {COCO_ANNOTATIONS_URL} and extract to {coco_dir}/"
        )
    with open(ann_file) as f:
        data = _json.load(f)

    # Group captions by image_id, take the first one
    from collections import defaultdict
    caps_by_img = defaultdict(list)
    for ann in data["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"])

    # Sort by image_id for determinism
    image_ids = sorted(caps_by_img.keys())
    prompts = [(img_id, caps_by_img[img_id][0]) for img_id in image_ids]
    return prompts[:NUM_IMAGES]


# ---------------------------------------------------------------------------
# APG (Adaptive Projected Guidance) — Sadat et al.
# ---------------------------------------------------------------------------
def apg_guidance(noise_pred_uncond, noise_pred_text, guidance_scale):
    """APG: project delta onto perpendicular component relative to uncond."""
    delta = noise_pred_text - noise_pred_uncond
    # Flatten for dot product
    B = delta.shape[0]
    delta_flat = delta.view(B, -1)
    uncond_flat = noise_pred_uncond.view(B, -1)

    # Parallel component: proj of delta onto uncond direction
    dot = (delta_flat * uncond_flat).sum(dim=-1, keepdim=True)
    norm_sq = (uncond_flat * uncond_flat).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    parallel = (dot / norm_sq) * uncond_flat

    # Perpendicular component
    perp = delta_flat - parallel
    perp = perp.view_as(delta)

    return noise_pred_uncond + guidance_scale * perp


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def load_pipeline(device, dtype):
    from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    eval_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "prompt_cache", "_eval_prompts.pt")
    if os.path.exists(eval_cache):
        print(f"Eval prompt cache found; loading without text encoders.", flush=True)
        pipeline = MyStableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype, token=hf_token,
            text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        )
        pipeline.transformer.to(device)
        pipeline.vae.to(device)
    else:
        pipeline = MyStableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype, token=hf_token,
        )
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    return pipeline


def _align_eval_conditioning(
    pipeline,
    device,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
):
    from src.pipelines.my_pipeline_stable_diffusion3 import _align_transformer_conditioning

    _, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
        _align_transformer_conditioning(
            pipeline.transformer,
            torch.device(device),
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
    )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


class AutoLambdaWrapper(torch.nn.Module):
    """Drop-in wrapper that replaces fixed lambda with geometry-based lambda_t."""
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, timestep, l, noise_pred_uncond, noise_pred_text, **mlp_extras):
        import torch.nn.functional as F
        v_u = noise_pred_uncond
        delta_t = noise_pred_text - noise_pred_uncond
        B = v_u.shape[0]
        cos_sim = F.cosine_similarity(v_u.reshape(B, -1), delta_t.reshape(B, -1), dim=1)
        x = torch.clamp((1.0 + cos_sim) / 2.0, 0.0, 1.0)
        lambda_t = 0.5 + torch.sign(x - 0.5) * torch.sqrt(torch.abs(2.0 * x - 1.0)) / 2.0
        return self.mlp(timestep, lambda_t, noise_pred_uncond, noise_pred_text, **mlp_extras)


def load_guidance_model(checkpoint_path, device):
    from src.model.guidance_scale_model import ScalarMLP
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get('model_config') \
        or checkpoint.get('config', {}).get('guidance_scale_model', {})
    state_dict = checkpoint.get('model_state_dict') \
        or checkpoint.get('guidance_scale_model')
    model = ScalarMLP(
        hidden_size=model_cfg.get('hidden_size', 128),
        output_size=model_cfg.get('output_size', 1),
        n_layers=model_cfg.get('n_layers', 2),
        t_embed_dim=model_cfg.get('t_embed_dim', 4),
        delta_embed_dim=model_cfg.get('delta_embed_dim', 4),
        lambda_embed_dim=model_cfg.get('lambda_embed_dim', 4),
        interval_embed_dim=model_cfg.get('interval_embed_dim', 0),
        c_embed_dim=model_cfg.get('c_embed_dim', 0),
        c_input_dim=model_cfg.get('c_input_dim', 2048),
        t_embed_normalization=model_cfg.get('t_embed_normalization', 1e3),
        num_timesteps=model_cfg.get('num_timesteps') or checkpoint.get('config', {}).get('diffusion', {}).get('num_sampling_steps') or checkpoint.get('config', {}).get('diffusion', {}).get('num_timesteps'),
        delta_embed_normalization=model_cfg.get('delta_embed_normalization', 5.0),
        w_bias=model_cfg.get('w_bias', 1.0),
        w_scale=model_cfg.get('w_scale', 1.0),
    ).to(device, dtype=torch.float32)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    diff_cfg = checkpoint.get('config', {}).get('diffusion', {})
    training_num_timesteps = diff_cfg.get('num_sampling_steps') or diff_cfg.get('num_timesteps')
    is_extended = model_cfg.get('interval_embed_dim', 0) > 0 or model_cfg.get('c_embed_dim', 0) > 0
    return model, training_num_timesteps, is_extended


def generate_single(pipeline, prompt, seed, device, guidance_scale,
                    use_cfgpp=False, use_apg=False,
                    guidance_scale_model=None, guidance_lambda=None,
                    use_fsg=False, fsg_iterations=3, cached_embeds=None):
    """Generate a single image. Supports CFG, CFG++, APG, Annealing, and FSG."""
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    use_annealing = guidance_scale_model is not None and guidance_lambda is not None

    if use_apg:
        with torch.inference_mode():
            return _generate_apg(pipeline, prompt, seed, device, guidance_scale, generator,
                                 cached_embeds=cached_embeds)

    kwargs = dict(
        guidance_scale=guidance_scale,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        use_annealing_guidance=use_annealing,
        use_cfgpp=use_cfgpp,
    )
    if cached_embeds is not None:
        pe, npe, ppe, nppe = _align_eval_conditioning(pipeline, device, *cached_embeds)
        kwargs["prompt_embeds"] = pe
        kwargs["negative_prompt_embeds"] = npe
        kwargs["pooled_prompt_embeds"] = ppe
        kwargs["negative_pooled_prompt_embeds"] = nppe
    else:
        kwargs["prompt"] = prompt
    if use_annealing:
        kwargs["guidance_scale_model"] = guidance_scale_model
        kwargs["guidance_lambda"] = guidance_lambda
        if use_fsg:
            kwargs["use_fsg"] = True
            kwargs["fsg_iterations"] = fsg_iterations

    with torch.inference_mode():
        return pipeline(**kwargs).images[0]


def _generate_apg(pipeline, prompt, seed, device, guidance_scale, generator, cached_embeds=None):
    """Generate with APG by temporarily patching the pipeline's guidance step."""
    if cached_embeds is not None:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            cached_embeds
    else:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None,
            do_classifier_free_guidance=True,
        )
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
        _align_eval_conditioning(
            pipeline,
            device,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
    )

    # Prepare timesteps
    pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipeline.scheduler.timesteps

    # Prepare latents
    num_channels_latents = pipeline.transformer.config.in_channels
    shape = (1, num_channels_latents,
             pipeline.default_sample_size, pipeline.default_sample_size)
    latents = torch.randn(shape, generator=generator, device=device,
                          dtype=pipeline.transformer.dtype)
    # Flow-matching schedulers (SD3) don't use init_noise_sigma
    if hasattr(pipeline.scheduler, 'init_noise_sigma'):
        latents = latents * pipeline.scheduler.init_noise_sigma

    # Denoising loop with APG (sequential uncond/cond to save memory)
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0])

        noise_pred_uncond = pipeline.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=negative_prompt_embeds,
            pooled_projections=negative_pooled_prompt_embeds,
            return_dict=False,
        )[0]

        noise_pred_text = pipeline.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # APG: perpendicular guidance
        noise_pred = apg_guidance(noise_pred_uncond, noise_pred_text, guidance_scale)

        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        del noise_pred_uncond, noise_pred_text, noise_pred

    # Decode
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    return image


def generate_images_for_config(pipeline, prompts, save_dir, device,
                               guidance_scale, use_cfgpp=False, use_apg=False,
                               guidance_scale_model=None, guidance_lambda=None,
                               use_fsg=False, fsg_iterations=3,
                               rank=0, world_size=1, prompt_embed_cache=None):
    """Generate images for all prompts, saving to save_dir/. Skip existing."""
    os.makedirs(save_dir, exist_ok=True)

    # Shard work across GPUs
    my_prompts = [(i, p) for i, p in enumerate(prompts) if i % world_size == rank]

    for idx, (image_id, caption) in tqdm(my_prompts, desc=os.path.basename(save_dir),
                                          disable=(rank != 0)):
        img_path = os.path.join(save_dir, f"{image_id:012d}.png")
        if os.path.exists(img_path):
            continue
        img = generate_single(
            pipeline, caption, image_id + SEED_OFFSET, device,
            guidance_scale, use_cfgpp=use_cfgpp, use_apg=use_apg,
            guidance_scale_model=guidance_scale_model, guidance_lambda=guidance_lambda,
            use_fsg=use_fsg, fsg_iterations=fsg_iterations,
            cached_embeds=prompt_embed_cache.get(caption) if prompt_embed_cache else None,
        )
        img.save(img_path)
        del img
        if idx % 50 == 0:
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_fid(gen_dir, ref_dir):
    """Compute FID between generated images and COCO val2017 reference."""
    try:
        from cleanfid import fid as cleanfid
        score = cleanfid.compute_fid(gen_dir, ref_dir)
        return score
    except ImportError:
        # Fallback: use torch-fidelity or pytorch-fid
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths
            score = calculate_fid_given_paths(
                [gen_dir, ref_dir], batch_size=64, device="cuda",
                dims=2048, num_workers=4)
            return score
        except ImportError:
            print("WARNING: Neither clean-fid nor pytorch-fid installed. "
                  "Install with: pip install clean-fid")
            return float('nan')


def compute_clip_score(gen_dir, prompts):
    """Compute mean CLIP similarity between generated images and their prompts."""
    try:
        import clip
    except ImportError:
        try:
            import open_clip as clip
        except ImportError:
            print("WARNING: CLIP not installed. Install with: pip install openai-clip")
            return float('nan')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try openai CLIP first
    try:
        model, preprocess = clip.load("ViT-L/14", device=device)
        tokenize = clip.tokenize
    except (AttributeError, Exception):
        # open_clip interface
        model, _, preprocess = clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        model = model.to(device)
        tokenize = clip.get_tokenizer('ViT-L-14')

    scores = []
    for image_id, caption in tqdm(prompts, desc="CLIP score"):
        img_path = os.path.join(gen_dir, f"{image_id:012d}.png")
        if not os.path.exists(img_path):
            continue
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text = tokenize([caption]).to(device) if callable(tokenize) else tokenize([caption]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = (image_features * text_features).sum(dim=-1).item()
        scores.append(sim)

    return np.mean(scores) if scores else float('nan')


def compute_image_reward(gen_dir, prompts):
    """Compute mean ImageReward score."""
    try:
        import ImageReward as IR
        model = IR.load("ImageReward-v1.0")
    except ImportError:
        print("WARNING: ImageReward not installed. Install with: pip install image-reward")
        return float('nan')

    scores = []
    for image_id, caption in tqdm(prompts, desc="ImageReward"):
        img_path = os.path.join(gen_dir, f"{image_id:012d}.png")
        if not os.path.exists(img_path):
            continue
        score = model.score(caption, img_path)
        scores.append(score)

    return np.mean(scores) if scores else float('nan')


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------
def get_rank_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate FID/CLIP/ImageReward (COCO 5k)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to annealing guidance checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results (e.g. results/images/$SLURM_JOB_ID)")
    parser.add_argument("--coco_dir", type=str,
                        default=os.path.join(_REPO_ROOT, "data", "coco2017"),
                        help="Path to COCO 2017 dataset")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for this run (used in CSV output)")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip baseline generation (assume cached)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip all generation (only compute metrics)")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip metrics computation (only generate images)")
    parser.add_argument("--baseline_indices", type=int, nargs="+", default=None,
                        help="Indices into BASELINE_CONFIGS to generate (0-based). "
                             "If not set, generate all. Use to split work across jobs.")
    parser.add_argument("--annealing_lambdas", type=float, nargs="+",
                        default=ANNEALING_LAMBDAS,
                        help="Lambda values to evaluate")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of inference steps (must match training num_timesteps)")
    args = parser.parse_args()

    global NUM_INFERENCE_STEPS
    if args.num_steps is not None:
        NUM_INFERENCE_STEPS = args.num_steps

    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dtype = torch.float16

    if rank != 0:
        import builtins
        builtins.print = lambda *a, **kw: None

    print(f"[GPU {world_size}x] Device: {device} — {torch.cuda.get_device_name(local_rank)}")

    # --- Load COCO prompts ---
    prompts = get_coco_prompts(args.coco_dir)
    print(f"Loaded {len(prompts)} COCO prompts")

    if args.checkpoint and args.num_steps is None:
        checkpoint_meta = torch.load(args.checkpoint, map_location="cpu")
        diff_cfg = checkpoint_meta.get("config", {}).get("diffusion", {})
        inferred_steps = diff_cfg.get("num_sampling_steps") or diff_cfg.get("num_timesteps")
        if inferred_steps is not None:
            NUM_INFERENCE_STEPS = inferred_steps
            print(f"Auto-inferred num_inference_steps={NUM_INFERENCE_STEPS} from checkpoint")

    coco_val_dir = os.path.join(args.coco_dir, "val2017")
    os.makedirs(args.output_dir, exist_ok=True)
    BASELINE_CACHE_DIR = os.path.join(_BASELINE_CACHE_ROOT, f"steps{NUM_INFERENCE_STEPS}_n{NUM_IMAGES}")
    os.makedirs(BASELINE_CACHE_DIR, exist_ok=True)

    # Resolve which baselines this job handles
    if args.baseline_indices is not None:
        my_baselines = [BASELINE_CONFIGS[i] for i in args.baseline_indices
                        if i < len(BASELINE_CONFIGS)]
        print(f"Baseline subset: indices {args.baseline_indices} → "
              f"{[b[0] for b in my_baselines]}")
    else:
        my_baselines = list(BASELINE_CONFIGS)

    if not args.skip_generation:
        # --- Load pipeline ---
        pipeline = load_pipeline(device, dtype)

        # --- Load eval prompt cache if available ---
        eval_cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "prompt_cache", "_eval_prompts.pt")
        eval_prompt_cache = torch.load(eval_cache_path, map_location="cpu") if os.path.exists(eval_cache_path) else None
        if eval_prompt_cache:
            print(f"Loaded {len(eval_prompt_cache)} cached eval prompts.", flush=True)

        # --- Generate baselines (cached) ---
        if not args.skip_baselines:
            print("\n=== Generating baselines (cached) ===")
            for dir_name, method, w, use_cfgpp, use_apg in my_baselines:
                save_dir = os.path.join(BASELINE_CACHE_DIR, dir_name)
                n_existing = len([f for f in os.listdir(save_dir) if f.endswith('.png')]) \
                    if os.path.exists(save_dir) else 0
                if n_existing >= NUM_IMAGES:
                    print(f"  {dir_name}: {n_existing} images cached, skipping")
                    continue
                print(f"  {dir_name}: generating ({n_existing}/{NUM_IMAGES} cached)...")
                generate_images_for_config(
                    pipeline, prompts, save_dir, device,
                    guidance_scale=w, use_cfgpp=use_cfgpp, use_apg=use_apg,
                    rank=rank, world_size=world_size,
                    prompt_embed_cache=eval_prompt_cache,
                )

        # --- Generate annealing images ---
        if args.checkpoint:
            guidance_model, training_steps, is_extended = load_guidance_model(args.checkpoint, device)
            if args.num_steps is None and training_steps is not None:
                NUM_INFERENCE_STEPS = training_steps
                print(f"  Auto-inferred num_inference_steps={NUM_INFERENCE_STEPS} from checkpoint")

            # Build the annealing variants to evaluate
            # Always: 4 fixed lambdas + auto-lambda (regular inference)
            # If extended MLP (FSG-trained): also add 4 lambdas + auto with FSG inference
            auto_model = AutoLambdaWrapper(guidance_model).to(device).eval()
            variants = []
            for lam in args.annealing_lambdas:
                variants.append((f"annealing_lambda{lam:.2f}", guidance_model, lam, False))
            variants.append(("annealing_auto_lambda", auto_model, 0.0, False))
            if is_extended:
                print("  Detected extended MLP — adding FSG inference variants")
                for lam in args.annealing_lambdas:
                    variants.append((f"annealing_fsg_lambda{lam:.2f}", guidance_model, lam, True))
                variants.append(("annealing_fsg_auto_lambda", auto_model, 0.0, True))

            print(f"\n=== Generating annealing images ({len(variants)} variants) ===")
            for dir_name, model_to_use, lam, use_fsg in variants:
                save_dir = os.path.join(args.output_dir, dir_name)
                n_existing = len([f for f in os.listdir(save_dir) if f.endswith('.png')]) \
                    if os.path.exists(save_dir) else 0
                if n_existing >= NUM_IMAGES:
                    print(f"  {dir_name}: {n_existing} images exist, skipping")
                    continue
                print(f"  {dir_name}: generating ({n_existing}/{NUM_IMAGES} cached)...")
                generate_images_for_config(
                    pipeline, prompts, save_dir, device,
                    guidance_scale=ANNEALING_GUIDANCE_SCALE,
                    guidance_scale_model=model_to_use, guidance_lambda=lam,
                    use_fsg=use_fsg,
                    rank=rank, world_size=world_size,
                    prompt_embed_cache=eval_prompt_cache,
                )

        # Synchronize GPUs before metrics
        if world_size > 1:
            torch.distributed.init_process_group(backend="nccl")
            torch.distributed.barrier()

        # Free pipeline memory before metrics
        del pipeline
        torch.cuda.empty_cache()

    # --- Compute metrics (rank 0 only) ---
    if rank == 0 and not args.skip_metrics:
        print("\n=== Computing metrics ===")
        results = []

        # Baselines (only compute metrics for configs this job handled)
        for dir_name, method, w, use_cfgpp, use_apg in my_baselines:
            gen_dir = os.path.join(BASELINE_CACHE_DIR, dir_name)
            if not os.path.exists(gen_dir):
                print(f"  {dir_name}: no images found, skipping")
                continue
            print(f"\n  Computing metrics for {dir_name}...")
            fid = compute_fid(gen_dir, coco_val_dir)
            clip_score = compute_clip_score(gen_dir, prompts)
            img_reward = compute_image_reward(gen_dir, prompts)
            results.append({
                "method": method,
                "config": dir_name,
                "guidance_scale": w,
                "lambda": "",
                "FID": f"{fid:.2f}",
                "CLIP": f"{clip_score:.4f}",
                "ImageReward": f"{img_reward:.4f}",
            })
            print(f"    FID={fid:.2f}  CLIP={clip_score:.4f}  ImageReward={img_reward:.4f}")

        # Annealing (regular + auto, plus FSG variants if extended MLP)
        if args.checkpoint:
            label = args.label or os.path.basename(args.checkpoint)

            # Build the list of variants to score (matching what was generated above)
            anneal_variants = []
            for lam in args.annealing_lambdas:
                anneal_variants.append((f"annealing_lambda{lam:.2f}",
                                        f"Annealing ({label})", lam))
            anneal_variants.append(("annealing_auto_lambda",
                                    f"Annealing auto-λ ({label})", "auto"))
            # Detect extended MLP from checkpoint to also score FSG variants
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            mcfg = ckpt.get('model_config') or ckpt.get('config', {}).get('guidance_scale_model', {})
            is_extended = mcfg.get('interval_embed_dim', 0) > 0 or mcfg.get('c_embed_dim', 0) > 0
            del ckpt
            if is_extended:
                for lam in args.annealing_lambdas:
                    anneal_variants.append((f"annealing_fsg_lambda{lam:.2f}",
                                            f"Annealing FSG ({label})", lam))
                anneal_variants.append(("annealing_fsg_auto_lambda",
                                        f"Annealing FSG auto-λ ({label})", "auto"))

            for dir_name, method_label, lam in anneal_variants:
                gen_dir = os.path.join(args.output_dir, dir_name)
                if not os.path.exists(gen_dir):
                    print(f"  {dir_name}: no images found, skipping")
                    continue
                print(f"\n  Computing metrics for {dir_name}...")
                fid = compute_fid(gen_dir, coco_val_dir)
                clip_score = compute_clip_score(gen_dir, prompts)
                img_reward = compute_image_reward(gen_dir, prompts)
                results.append({
                    "method": method_label,
                    "config": dir_name,
                    "guidance_scale": ANNEALING_GUIDANCE_SCALE,
                    "lambda": lam,
                    "FID": f"{fid:.2f}",
                    "CLIP": f"{clip_score:.4f}",
                    "ImageReward": f"{img_reward:.4f}",
                })
                print(f"    FID={fid:.2f}  CLIP={clip_score:.4f}  ImageReward={img_reward:.4f}")

        # --- Save CSV ---
        csv_path = os.path.join(args.output_dir, "metrics_table.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        if results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n=== Results saved to {csv_path} ===")

            # Also print as formatted table
            print("\n" + "=" * 80)
            print(f"{'Method':<20} {'Config':<20} {'w':>5} {'λ':>5} {'FID':>8} {'CLIP':>8} {'ImgRew':>10}")
            print("-" * 80)
            for r in results:
                print(f"{r['method']:<20} {r['config']:<20} {r['guidance_scale']:>5} "
                      f"{str(r['lambda']):>5} {r['FID']:>8} {r['CLIP']:>8} {r['ImageReward']:>10}")
            print("=" * 80)

        # Save as JSON too
        json_path = os.path.join(args.output_dir, "metrics_table.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Append to cumulative results file
        global_csv = os.path.join(os.path.dirname(args.output_dir), "results.csv")
        write_header = not os.path.exists(global_csv)
        ckpt_name = os.path.basename(args.checkpoint) if args.checkpoint else ""
        with open(global_csv, "a", newline="") as f:
            fields = ["run", "checkpoint"] + list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow({"run": os.path.basename(args.output_dir), "checkpoint": ckpt_name, **r})


if __name__ == "__main__":
    main()
