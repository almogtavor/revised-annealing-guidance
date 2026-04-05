"""Batch SD3 image generation for all prompts across lambda values."""
import os
import sys
import json
import torch
from PIL import Image
import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))
import src.utils.wandb_utils as wb

# All prompts organized by figure
PROMPTS_BY_FIGURE = {
    "fig_1": [
        (1, "A man juggles flaming hats."),
        (2, "Two giraffes in astronaut suits repairing a spacecraft on Mars."),
        (3, "A dragon and a knight playing cards in a tavern."),
        (4, "A robot painting a portrait."),
        (5, "A cake with onions on top of it."),
        (6, "A photo of a ballerina flamingo dancing on the beach."),
        (7, "A donkey in a clown costume giving a lecture at the front of a lecture hall. There are many students in the lecture hall."),
        (8, "A mid-air dog practicing karate in a Japanese dojo, wearing a white gi with a black belt on wooden floors."),
        (9, "A man is seated on a wooden stool against a white background, dressed in a blue suit with a tie and brown shoes."),
    ],
    "fig_2": [
        (10, "Woman in black dress on the red carpet wearing a ring on the finger."),
        (11, "Two dogs, one cat."),
    ],
    "fig_5": [
        (12, "A photo of unicorn driving a jeep in the desert."),
        (13, "A knight in rainbow armor riding a dragon made of fire."),
        (14, "A cat looking through a glass of water."),
        (15, "A yellow dog runs to grab a yellow frisbee in the grass."),
        (16, "Bear cubs play among the fallen tree limbs."),
        (17, "A traffic sign that has a picture of a man holding a surfboard on it."),
    ],
    "fig_10": [
        (18, "A statue of Abraham Lincoln wearing an opaque and shiny astronaut's helmet. The statue sits on the moon."),
        (19, "A baby sitting on a female's lap staring into the camera."),
        (20, "A bride and groom cutting their wedding cake."),
        (21, "A small boy trying to fly a small kite."),
        (22, "Five red balls on a table."),
        (23, "A man and child next to a horse."),
        (24, "A demonic looking chucky-like doll standing next to a white clock."),
        (25, "Older woman hula hooping in backyard."),
        (26, "A dog running with a stick in its mouth, Eiffel tower in the background."),
        (27, "A girl riding a giant bird over a futuristic city."),
    ],
    "fig_15": [
        (28, "A woman bending over and looking inside of a mini fridge."),
        (29, "A coffee cup that is full of holes."),
        (30, "A cat that is standing looking through a glass."),
        (31, "Two giraffes moving very quickly in the woods."),
        (32, "A man standing next to an elephant next to his trunk."),
        (33, "An airplane leaving a trail in the sky."),
        (34, "A tiger dancing on a frozen lake."),
        (35, "A tropical bird."),
        (36, "A woman standing next to a young man near a pile of fruit."),
        (37, "A grand piano next to the net of a tennis court."),
    ],
    "fig_16": [
        (38, "A child in a yellow raincoat jumping into a puddle, holding a red balloon."),
        (39, "A white bear in glasses, wearing tuxedo, glowing hat, and with cigare at the British queen reception."),
        (40, "A man is shaking hands with another man."),
        (41, "A man stands beside his black and red motorcycle near a park."),
        (42, "An elephant with sunglasses plays with a flute."),
        (43, "Two samurai cats, katanas drawn, petals swirling in the background."),
        (44, "A horse in a field."),
        (45, "An owl delivering mail at a snowy train station."),
        (46, "A lavender backpack with a triceratops stuffed animal head on top."),
        (47, "A light blue bicycle chained to a pole in front of a red building."),
    ],
    "fig_17": [
        (48, "A giant snail race through the streets of an old European town, onlookers cheering, mid-day sun."),
        (49, "Photo of alpine ski resort with yeti instead of humans. It wears a red helmet."),
        (50, "A giant meteorite with the words hello people approaching the earth."),
        (51, "A man standing next to bikes and a motorcycle."),
        (52, "Long-exposure night shot of neon with a huge ghostly animal."),
        (53, "A present."),
        (54, "A ballet dancer next to a waterfall."),
        (55, "Bear drinking coffee on a sunny morning street, in Italy."),
        (56, "A man riding a motorcycle while eating food."),
        (57, "Two rams trying to solve a math equation."),
    ],
}

LAMBDA_VALUES = [0.0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
SEED = 1000
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0


# ---------------------------------------------------------------------------
# Multi-GPU helpers (no DDP — just split work across ranks via torchrun)
# ---------------------------------------------------------------------------

def _get_rank_info():
    """Return (rank, world_size, local_rank) from torchrun env vars.
    Falls back to (0, 1, 0) for single-GPU runs."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size, local_rank


def _shard_work(items, rank, world_size):
    """Return the slice of *items* that belongs to this rank."""
    return [x for i, x in enumerate(items) if i % world_size == rank]


def slugify(text, max_len=40):
    """Create a filesystem-safe slug from text."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text[:max_len].strip('_')


def create_grid(images, labels, save_path):
    """Create a grid of images with labels."""
    if not images:
        return

    n = len(images)
    img_w, img_h = images[0].size

    # Single row grid
    grid_w = n * img_w
    grid_h = img_h + 50  # space for labels

    grid = Image.new('RGB', (grid_w, grid_h), 'white')

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * img_w
        grid.paste(img, (x, 0))
        draw.text((x + 5, img_h + 5), label, fill='black', font=font)

    grid.save(save_path)
    return grid


def create_figure_summary(fig_dir, fig_name, prompts, lambda_values, save_path):
    """Create a summary grid for an entire figure: rows=prompts, cols=lambdas."""
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Collect all images (rows=prompts, cols=lambdas)
    thumb_size = 256
    col_header_h = 60
    row_label_w = 420
    padding = 4

    n_rows = len(prompts)
    n_cols = len(lambda_values)

    grid_w = row_label_w + n_cols * (thumb_size + padding) + padding
    grid_h = col_header_h + n_rows * (thumb_size + padding) + padding

    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    # Draw column headers (lambda values + baselines)
    for j, lam in enumerate(lambda_values):
        x = row_label_w + j * (thumb_size + padding) + padding
        if isinstance(lam, tuple):
            # Baseline: (dir_name, label)
            label = lam[1]
        elif lam == 0.0:
            label = "λ=0 (vanilla)"
        elif lam == 1.0:
            label = f"λ={lam:.1f}"
        else:
            label = f"λ={lam:.2f}"
        draw.text((x + 10, 10), label, fill='black', font=font)

    # Draw each row
    for i, (prompt_id, prompt_text) in enumerate(prompts):
        y = col_header_h + i * (thumb_size + padding) + padding
        prompt_slug = slugify(prompt_text)
        prompt_dir_name = f"prompt_{prompt_id:03d}_{prompt_slug}"
        prompt_dir = os.path.join(fig_dir, prompt_dir_name)

        # Row label (prompt text, wrapped)
        short_text = prompt_text if len(prompt_text) <= 55 else prompt_text[:52] + "..."
        draw.text((5, y + 5), f"#{prompt_id}", fill='black', font=font)
        draw.text((5, y + 40), short_text, fill='gray', font=font_small)

        for j, lam in enumerate(lambda_values):
            x = row_label_w + j * (thumb_size + padding) + padding
            if isinstance(lam, tuple):
                # Baseline: (dir_name, label)
                img_path = os.path.join(prompt_dir, lam[0], f"seed_{SEED}.png")
            else:
                img_path = os.path.join(prompt_dir, f"lambda_{lam:.2f}", f"seed_{SEED}.png")

            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
                grid.paste(img, (x, y))
            else:
                # Draw placeholder
                draw.rectangle([x, y, x + thumb_size, y + thumb_size], fill='lightgray', outline='gray')
                draw.text((x + 20, y + thumb_size // 2), "missing", fill='gray', font=font_small)

    grid.save(save_path, quality=95)
    print(f"  Figure summary saved: {save_path}")
    return grid


def load_pipeline_and_model(checkpoint_path, device, dtype):
    """Load SD3 pipeline and guidance scale model."""
    from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
    from src.model.guidance_scale_model import ScalarMLP

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    print("Loading SD3 pipeline...")
    pipeline = MyStableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=dtype,
        token=hf_token,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    print(f"Loading guidance model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both old format (guidance_scale_model / config.guidance_scale_model)
    # and new format (model_state_dict / model_config)
    model_cfg = checkpoint.get('model_config') \
        or checkpoint.get('config', {}).get('guidance_scale_model', {})
    state_dict = checkpoint.get('model_state_dict') \
        or checkpoint.get('guidance_scale_model')

    # Always load guidance model in float32 (trained in float32, small network)
    guidance_scale_model = ScalarMLP(
        hidden_size=model_cfg.get('hidden_size', 128),
        output_size=model_cfg.get('output_size', 1),
        n_layers=model_cfg.get('n_layers', 2),
        t_embed_dim=model_cfg.get('t_embed_dim', 4),
        delta_embed_dim=model_cfg.get('delta_embed_dim', 4),
        lambda_embed_dim=model_cfg.get('lambda_embed_dim', 4),
        t_embed_normalization=model_cfg.get('t_embed_normalization', 1e3),
        delta_embed_normalization=model_cfg.get('delta_embed_normalization', 5.0),
        w_bias=model_cfg.get('w_bias', 1.0),
        w_scale=model_cfg.get('w_scale', 1.0),
    ).to(device, dtype=torch.float32)

    guidance_scale_model.load_state_dict(state_dict, strict=True)
    guidance_scale_model.eval()

    # Verify the checkpoint is not corrupted (training may have diverged)
    for name, param in guidance_scale_model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(
                f"Guidance model has NaN/Inf weights in '{name}'. "
                f"Training likely diverged. Use an earlier checkpoint."
            )

    return pipeline, guidance_scale_model


def generate_image(pipeline, guidance_scale_model, prompt, lambda_val, seed, device,
                    cached_embeds=None):
    """Generate a single image."""
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    kwargs = dict(
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        use_annealing_guidance=True,
        guidance_scale_model=guidance_scale_model,
        guidance_lambda=lambda_val,
    )
    if cached_embeds:
        kwargs.update(prompt_embeds=cached_embeds[0].to(device),
                      negative_prompt_embeds=cached_embeds[1].to(device),
                      pooled_prompt_embeds=cached_embeds[2].to(device),
                      negative_pooled_prompt_embeds=cached_embeds[3].to(device))
    else:
        kwargs["prompt"] = prompt

    return pipeline(**kwargs).images[0]


def generate_baseline(pipeline, prompt, seed, device, guidance_scale,
                      use_cfgpp=False, cached_embeds=None):
    """Generate a single image using standard CFG or CFG++ with a fixed guidance scale."""
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    kwargs = dict(
        guidance_scale=guidance_scale,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        use_annealing_guidance=False,
        use_cfgpp=use_cfgpp,
    )
    if cached_embeds:
        kwargs.update(prompt_embeds=cached_embeds[0].to(device),
                      negative_prompt_embeds=cached_embeds[1].to(device),
                      pooled_prompt_embeds=cached_embeds[2].to(device),
                      negative_pooled_prompt_embeds=cached_embeds[3].to(device))
    else:
        kwargs["prompt"] = prompt

    return pipeline(**kwargs).images[0]


# Baseline configurations: (dir_name, label, guidance_scale, use_cfgpp)
BASELINES = [
    ("cfg_w3.5",  "CFG w=3.5",  3.5,  False),
    ("cfg_w7",    "CFG w=7",    7.0,  False),
    ("cfg_w12",   "CFG w=12",   12.0, False),
    ("cfgpp_w08", "CFG++ w=0.8", 0.8, True),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='src/model/checkpoints/checkpoint.pt',
                        help='Path to guidance model checkpoint')
    parser.add_argument('--checkpoint_id', type=str, default='checkpoint_id8',
                        help='Checkpoint identifier for output directory')
    parser.add_argument('--output_root', type=str, default='results/images',
                        help='Root output directory')
    parser.add_argument('--figures', type=str, nargs='+', default=None,
                        help='Specific figures to generate (e.g., fig_1 fig_5). Default: all')
    parser.add_argument('--prompts', type=int, nargs='+', default=None,
                        help='Specific prompt IDs to generate. Default: all')
    parser.add_argument('--lambdas', type=float, nargs='+', default=None,
                        help='Specific lambda values. Default: 0.0 0.2 0.4 0.5 0.6 0.8 1.0')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing images instead of skipping')
    parser.add_argument('--baselines', action='store_true',
                        help='Also generate baseline images (CFG w=7, CFG w=12, CFG++ w=0.8)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. Refusing to run on CPU (float16 will hang).")
        sys.exit(1)

    rank, world_size, local_rank = _get_rank_info()
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    dtype = torch.float16

    # Suppress prints on non-main ranks
    if rank != 0:
        import builtins
        builtins.print = lambda *a, **kw: None

    print(f"[GPU {world_size}x] Device: {device} — {torch.cuda.get_device_name(local_rank)}")

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_REPO_ROOT, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    pipeline, guidance_scale_model = load_pipeline_and_model(checkpoint_path, device, dtype)
    if rank == 0:
        wb.init_sampling(vars(args), guidance_scale_model)

    # Pre-encode all unique prompts once (avoids re-running T5-XXL per lambda)
    all_unique_prompts = list({p for prompts in PROMPTS_BY_FIGURE.values() for _, p in prompts})
    print(f"Pre-encoding {len(all_unique_prompts)} unique prompts...")
    prompt_embed_cache = {}
    with torch.no_grad():
        for i, p in enumerate(all_unique_prompts):
            pe, npe, ppe, nppe = pipeline.encode_prompt(
                prompt=p, prompt_2=None, prompt_3=None,
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
            prompt_embed_cache[p] = (pe.cpu(), npe.cpu(), ppe.cpu(), nppe.cpu())
            if (i + 1) % 10 == 0:
                print(f"  Encoded {i+1}/{len(all_unique_prompts)} prompts")
    print(f"Prompt encoding done. Freeing T5 encoder...")
    del pipeline.text_encoder_3
    torch.cuda.empty_cache()

    # Determine output root
    output_root = os.path.join(_REPO_ROOT, args.output_root, args.checkpoint_id)
    os.makedirs(output_root, exist_ok=True)

    # Determine which figures and lambdas to process
    figures_to_process = args.figures if args.figures else list(PROMPTS_BY_FIGURE.keys())
    lambda_values = args.lambdas if args.lambdas else LAMBDA_VALUES
    prompt_filter = set(args.prompts) if args.prompts else None

    # Build flat work list: [(fig_name, prompt_id, prompt_text, task_type, task_info), ...]
    # task_type: "lambda" or "baseline"
    # task_info: lambda_val (float) or (bl_dir_name, bl_label, bl_w, bl_cfgpp) tuple
    all_work = []
    for fig_name in figures_to_process:
        if fig_name not in PROMPTS_BY_FIGURE:
            continue
        for prompt_id, prompt_text in PROMPTS_BY_FIGURE[fig_name]:
            if prompt_filter and prompt_id not in prompt_filter:
                continue
            for lam in lambda_values:
                all_work.append((fig_name, prompt_id, prompt_text, "lambda", lam))
            if args.baselines:
                for bl in BASELINES:
                    all_work.append((fig_name, prompt_id, prompt_text, "baseline", bl))

    # Shard work across GPUs
    my_work = _shard_work(all_work, rank, world_size)
    total_expected = len(my_work)
    image_counter = 0
    total_images = 0
    start_time = datetime.datetime.now()

    print(f"\n{'='*60}")
    print(f"Batch Generation Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {output_root}")
    print(f"Total work items: {len(all_work)} across {world_size} GPU(s)")
    print(f"Lambdas: {lambda_values}")
    print(f"Seed: {SEED}")
    print(f"{'='*60}\n")

    for fig_name, prompt_id, prompt_text, task_type, task_info in my_work:
        fig_dir = os.path.join(output_root, fig_name)
        os.makedirs(fig_dir, exist_ok=True)
        prompt_slug = slugify(prompt_text)
        prompt_dir_name = f"prompt_{prompt_id:03d}_{prompt_slug}"
        prompt_dir = os.path.join(fig_dir, prompt_dir_name)
        os.makedirs(prompt_dir, exist_ok=True)

        if task_type == "lambda":
            lam = task_info
            lambda_dir = os.path.join(prompt_dir, f"lambda_{lam:.2f}")
            os.makedirs(lambda_dir, exist_ok=True)
            image_path = os.path.join(lambda_dir, f"seed_{SEED}.png")
            meta_path = os.path.join(lambda_dir, "meta.json")

            if os.path.exists(image_path) and not args.force:
                print(f"  prompt {prompt_id} λ={lam:.2f} - exists, skipping")
                image_counter += 1
                continue

            img_t0 = datetime.datetime.now()
            try:
                image = generate_image(
                    pipeline, guidance_scale_model,
                    prompt_text, lam, SEED, device,
                    cached_embeds=prompt_embed_cache.get(prompt_text),
                )
                image.save(image_path)
                meta = {
                    "prompt_id": prompt_id, "prompt": prompt_text,
                    "lambda": lam, "seed": SEED,
                    "guidance_scale": GUIDANCE_SCALE,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "checkpoint": args.checkpoint,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                total_images += 1
                image_counter += 1
                img_sec = (datetime.datetime.now() - img_t0).total_seconds()
                print(f"  prompt {prompt_id} λ={lam:.2f} | {img_sec:.1f}s | {image_counter}/{total_expected}", flush=True)
            except Exception as e:
                print(f"  [R{rank}] ERROR prompt {prompt_id} λ={lam:.2f}: {e}")
                image_counter += 1

        else:  # baseline
            bl_dir_name, bl_label, bl_w, bl_cfgpp = task_info
            bl_dir = os.path.join(prompt_dir, bl_dir_name)
            os.makedirs(bl_dir, exist_ok=True)
            bl_image_path = os.path.join(bl_dir, f"seed_{SEED}.png")
            bl_meta_path = os.path.join(bl_dir, "meta.json")

            if os.path.exists(bl_image_path) and not args.force:
                print(f"  prompt {prompt_id} {bl_label} - exists, skipping")
                image_counter += 1
                continue

            bl_t0 = datetime.datetime.now()
            try:
                bl_img = generate_baseline(
                    pipeline, prompt_text, SEED, device,
                    guidance_scale=bl_w, use_cfgpp=bl_cfgpp,
                    cached_embeds=prompt_embed_cache.get(prompt_text),
                )
                bl_img.save(bl_image_path)
                bl_meta = {
                    "prompt_id": prompt_id, "prompt": prompt_text,
                    "type": bl_dir_name, "label": bl_label, "seed": SEED,
                    "guidance_scale": bl_w, "use_cfgpp": bl_cfgpp,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                with open(bl_meta_path, 'w') as f:
                    json.dump(bl_meta, f, indent=2)
                total_images += 1
                image_counter += 1
                bl_sec = (datetime.datetime.now() - bl_t0).total_seconds()
                print(f"  prompt {prompt_id} {bl_label} | {bl_sec:.1f}s | {image_counter}/{total_expected}", flush=True)
            except Exception as e:
                print(f"  [R{rank}] ERROR prompt {prompt_id} {bl_label}: {e}")
                image_counter += 1

    # --- Barrier: wait for all ranks before summaries ---
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        dist.barrier()

    # Only rank 0 creates grids and summaries
    if rank == 0:
        all_prompt_data = []
        for fig_name in figures_to_process:
            if fig_name not in PROMPTS_BY_FIGURE:
                continue
            prompts = PROMPTS_BY_FIGURE[fig_name]
            fig_dir = os.path.join(output_root, fig_name)

            for prompt_id, prompt_text in prompts:
                if prompt_filter and prompt_id not in prompt_filter:
                    continue
                prompt_slug = slugify(prompt_text)
                prompt_dir_name = f"prompt_{prompt_id:03d}_{prompt_slug}"
                prompt_dir = os.path.join(fig_dir, prompt_dir_name)

                # Build grid from saved images
                images_for_grid, labels_for_grid = [], []
                for lam in lambda_values:
                    img_path = os.path.join(prompt_dir, f"lambda_{lam:.2f}", f"seed_{SEED}.png")
                    if os.path.exists(img_path):
                        images_for_grid.append(Image.open(img_path))
                        labels_for_grid.append(f"λ={lam:.2f}")
                if args.baselines:
                    for bl_dir_name, bl_label, _, _ in BASELINES:
                        img_path = os.path.join(prompt_dir, bl_dir_name, f"seed_{SEED}.png")
                        if os.path.exists(img_path):
                            images_for_grid.append(Image.open(img_path))
                            labels_for_grid.append(bl_label)

                if images_for_grid:
                    grid_path = os.path.join(prompt_dir, "grid.png")
                    create_grid(images_for_grid, labels_for_grid, grid_path)
                    print(f"  Grid saved: {grid_path}")

                all_prompt_data.append({
                    "prompt_id": prompt_id, "prompt": prompt_text,
                    "figure": fig_name, "directory": prompt_dir_name,
                })

            # Figure summary grid
            fig_prompts = [(pid, pt) for pid, pt in prompts
                           if prompt_filter is None or pid in prompt_filter]
            if fig_prompts:
                summary_lambdas = list(lambda_values) + ([(bl[0], bl[1]) for bl in BASELINES] if args.baselines else [])
                summary_path = os.path.join(fig_dir, f"{fig_name}_summary.png")
                create_figure_summary(fig_dir, fig_name, fig_prompts, summary_lambdas, summary_path)

        # Save prompt index
        summary_dir = os.path.join(output_root, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        index_path = os.path.join(summary_dir, "prompt_index.json")
        with open(index_path, 'w') as f:
            json.dump({
                "prompts": all_prompt_data,
                "lambda_values": lambda_values,
                "seed": SEED,
                "checkpoint": args.checkpoint,
                "generated_at": datetime.datetime.now().isoformat(),
            }, f, indent=2)

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Images generated: {total_images}")
    print(f"Duration: {duration}")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}\n")
    if rank == 0:
        wb.finish()

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
