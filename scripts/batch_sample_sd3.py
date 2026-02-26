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

    # Draw column headers (lambda values)
    for j, lam in enumerate(lambda_values):
        x = row_label_w + j * (thumb_size + padding) + padding
        label = f"λ={lam:.1f}" if lam == 0.0 or lam == 1.0 else f"λ={lam:.2f}"
        if lam == 0.0:
            label = "λ=0 (vanilla)"
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

    return pipeline, guidance_scale_model


def generate_image(pipeline, guidance_scale_model, prompt, lambda_val, seed, device):
    """Generate a single image."""
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipeline(
        prompt=prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        use_annealing_guidance=True,
        guidance_scale_model=guidance_scale_model,
        guidance_lambda=lambda_val,
    ).images[0]

    return image


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
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_REPO_ROOT, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    pipeline, guidance_scale_model = load_pipeline_and_model(checkpoint_path, device, dtype)
    wb.init_sampling(vars(args), guidance_scale_model)

    # Determine output root
    output_root = os.path.join(_REPO_ROOT, args.output_root, args.checkpoint_id)
    os.makedirs(output_root, exist_ok=True)

    # Determine which figures and lambdas to process
    figures_to_process = args.figures if args.figures else list(PROMPTS_BY_FIGURE.keys())
    lambda_values = args.lambdas if args.lambdas else LAMBDA_VALUES
    prompt_filter = set(args.prompts) if args.prompts else None
    total_expected = sum(len(lambda_values) for fn in figures_to_process if fn in PROMPTS_BY_FIGURE for pid, _ in PROMPTS_BY_FIGURE[fn] if prompt_filter is None or pid in prompt_filter)
    image_counter = 0

    # Track all generated images for summary
    all_prompt_data = []
    total_images = 0
    start_time = datetime.datetime.now()

    print(f"\n{'='*60}")
    print(f"Batch Generation Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {output_root}")
    print(f"Figures: {figures_to_process}")
    print(f"Lambdas: {lambda_values}")
    print(f"Seed: {SEED}")
    print(f"{'='*60}\n")

    for fig_name in figures_to_process:
        if fig_name not in PROMPTS_BY_FIGURE:
            print(f"Warning: Unknown figure {fig_name}, skipping...")
            continue

        prompts = PROMPTS_BY_FIGURE[fig_name]
        fig_dir = os.path.join(output_root, fig_name)
        os.makedirs(fig_dir, exist_ok=True)

        print(f"\n--- Processing {fig_name} ({len(prompts)} prompts) ---")

        for prompt_id, prompt_text in prompts:
            if prompt_filter and prompt_id not in prompt_filter:
                continue

            prompt_slug = slugify(prompt_text)
            prompt_dir_name = f"prompt_{prompt_id:03d}_{prompt_slug}"
            prompt_dir = os.path.join(fig_dir, prompt_dir_name)
            os.makedirs(prompt_dir, exist_ok=True)

            print(f"\n  Prompt {prompt_id}: {prompt_text[:50]}...")

            images_for_grid = []
            labels_for_grid = []

            for lam in lambda_values:
                lambda_dir = os.path.join(prompt_dir, f"lambda_{lam:.2f}")
                os.makedirs(lambda_dir, exist_ok=True)

                image_path = os.path.join(lambda_dir, f"seed_{SEED}.png")
                meta_path = os.path.join(lambda_dir, "meta.json")

                # Skip if already exists (unless --force)
                if os.path.exists(image_path) and not args.force:
                    print(f"    lambda={lam:.2f} - exists, skipping")
                    img = Image.open(image_path)
                    images_for_grid.append(img)
                    labels_for_grid.append(f"λ={lam:.2f}")
                    continue

                img_t0 = datetime.datetime.now()
                try:
                    image = generate_image(
                        pipeline, guidance_scale_model,
                        prompt_text, lam, SEED, device
                    )
                    image.save(image_path)

                    # Save metadata
                    meta = {
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "lambda": lam,
                        "seed": SEED,
                        "guidance_scale": GUIDANCE_SCALE,
                        "num_inference_steps": NUM_INFERENCE_STEPS,
                        "checkpoint": args.checkpoint,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    images_for_grid.append(image)
                    labels_for_grid.append(f"λ={lam:.2f}")
                    total_images += 1
                    image_counter += 1
                    img_sec = (datetime.datetime.now() - img_t0).total_seconds()
                    print(f"    λ={lam:.2f} | {NUM_INFERENCE_STEPS} steps in {img_sec:.1f}s ({img_sec/NUM_INFERENCE_STEPS:.3f}s/step) | image {image_counter}/{total_expected}", flush=True)

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            # Create grid for this prompt
            if images_for_grid:
                grid_path = os.path.join(prompt_dir, "grid.png")
                create_grid(images_for_grid, labels_for_grid, grid_path)
                print(f"    Grid saved: {grid_path}")

            all_prompt_data.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "figure": fig_name,
                "directory": prompt_dir_name,
            })

        # Create figure summary grid
        fig_prompts = [(pid, pt) for pid, pt in prompts
                       if prompt_filter is None or pid in prompt_filter]
        if fig_prompts:
            summary_path = os.path.join(fig_dir, f"{fig_name}_summary.png")
            create_figure_summary(fig_dir, fig_name, fig_prompts, lambda_values, summary_path)

    # Create summary directory
    summary_dir = os.path.join(output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Save prompt index
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
    print(f"Total images generated: {total_images}")
    print(f"Duration: {duration}")
    print(f"Output directory: {output_root}")
    print(f"Summary: {summary_dir}")
    print(f"{'='*60}\n")
    wb.finish()


if __name__ == "__main__":
    main()
