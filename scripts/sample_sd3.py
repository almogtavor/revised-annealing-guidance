"""SD3 image generation script with annealing guidance support."""
import os
import sys
import torch

# Allow running as `python scripts/sample_sd3.py` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if hf_token and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = hf_token

if not torch.cuda.is_available():
    print("FATAL: CUDA not available. Refusing to run on CPU (float16 will hang).")
    sys.exit(1)
device = 'cuda'
dtype = torch.float16

print(f"Using device: {device}")
print("Loading SD3 model...")

# Use our custom pipeline with annealing guidance support
from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline

pipeline = MyStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=dtype,
    token=hf_token,
)
pipeline.to(device)

if hasattr(pipeline, "enable_attention_slicing"):
    pipeline.enable_attention_slicing()

print("Model loaded successfully!")

# Sampling parameters
seed = 42
prompt = "A mid-air dog practicing karate in a Japanese dojo, wearing a white gi with a black belt"
guidance_scale = 7.0
num_inference_steps = 28

out_dir = 'samples_sd3'
os.makedirs(out_dir, exist_ok=True)

# ====================
# 1. Vanilla CFG
# ====================
print(f"\n[1/2] Generating vanilla CFG image...")
print(f"Prompt: '{prompt}'")

generator = torch.Generator(device=device).manual_seed(seed)
image_vanilla = pipeline(
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
    use_annealing_guidance=False,
).images[0]

output_path_vanilla = f'{out_dir}/sd3_vanilla_cfg.png'
image_vanilla.save(output_path_vanilla)
print(f"Vanilla CFG image saved to: {output_path_vanilla}")

# ====================
# 2. Annealing Guidance
# ====================
print(f"\n[2/2] Generating annealing guidance image...")

# Load guidance scale model if available
checkpoint_path = os.path.join(_REPO_ROOT, 'src/model/checkpoints/checkpoint.pt')
if os.path.exists(checkpoint_path):
    from src.model.guidance_scale_model import ScalarMLP

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = dict(
        checkpoint.get('model_config')
        or checkpoint.get('config', {}).get('guidance_scale_model', {})
    )
    model_cfg.setdefault(
        'num_timesteps',
        checkpoint.get('config', {}).get('diffusion', {}).get('num_sampling_steps')
        or checkpoint.get('config', {}).get('diffusion', {}).get('num_timesteps')
    )

    guidance_scale_model = ScalarMLP(**model_cfg).to(device)

    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('guidance_scale_model')
    guidance_scale_model.load_state_dict(state_dict)
    guidance_scale_model.eval()
    print(f"Loaded guidance scale model from {checkpoint_path}")

    # Annealing guidance parameters
    guidance_lambda = 0.4  # Controls quality/alignment trade-off [0, 1]

    generator = torch.Generator(device=device).manual_seed(seed)
    image_annealing = pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,  # fallback, not used when annealing is enabled
        num_inference_steps=num_inference_steps,
        generator=generator,
        use_annealing_guidance=True,
        guidance_scale_model=guidance_scale_model,
        guidance_lambda=guidance_lambda,
    ).images[0]

    output_path_annealing = f'{out_dir}/sd3_annealing_guidance.png'
    image_annealing.save(output_path_annealing)
    print(f"Annealing guidance image saved to: {output_path_annealing}")
else:
    print(f"Checkpoint not found at {checkpoint_path}, skipping annealing guidance test.")
    print("To test annealing guidance, train a model first or provide a checkpoint.")

print("\nDone!")
