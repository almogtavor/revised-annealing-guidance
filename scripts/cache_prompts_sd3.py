"""Cache SD3 prompt embeddings to disk.

Loads text encoders (no transformer/VAE), encodes all captions, saves as fp32 .pt.
Only encodes positive prompts — negative (empty string) embeddings are encoded once
and saved as _negative.pt in the cache root.

Supports multi-worker parallelism via RANK/WORLD_SIZE env vars (e.g. torchrun).

Usage:
    # Single process:
    python scripts/cache_prompts_sd3.py --image_root ... --cache_dir ...
    # 8 parallel workers:
    torchrun --nproc_per_node=8 scripts/cache_prompts_sd3.py --image_root ... --cache_dir ...
"""
import argparse
import os
import torch
from glob import glob
from tqdm import tqdm

from src.data.dataset import trim_to_77_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str,
                        default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Load text encoders in fp16 (outputs still saved as fp32)")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    # Collect all .txt caption files
    shard_folders = sorted(
        f for f in os.listdir(args.image_root)
        if os.path.isdir(os.path.join(args.image_root, f))
    )
    txt_paths = []
    for shard in shard_folders:
        shard_path = os.path.join(args.image_root, shard)
        txts = sorted(glob(os.path.join(shard_path, "*.txt")))
        txt_paths.extend(txts)

    # Filter out already-cached
    to_cache = []
    for txt_path in txt_paths:
        rel = os.path.relpath(txt_path, args.image_root)
        cache_path = os.path.join(args.cache_dir, rel.replace(".txt", ".pt"))
        if not os.path.exists(cache_path):
            to_cache.append((txt_path, cache_path))

    # Shard work across workers
    my_items = to_cache[rank::world_size]

    if rank == 0:
        print(f"Total: {len(txt_paths)} captions, to cache: {len(to_cache)}, "
              f"this worker (rank {rank}/{world_size}): {len(my_items)}", flush=True)

    if not my_items:
        print(f"[rank {rank}] Nothing to cache. Done.", flush=True)
        return

    from diffusers import StableDiffusion3Pipeline
    gpu_device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    load_dtype = torch.float16 if args.fp16 else torch.float32
    dtype_label = "fp16" if args.fp16 else "fp32"
    if rank == 0:
        print(f"Loading SD3 text encoders ({dtype_label}, no transformer/VAE)...", flush=True)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=load_dtype, token=hf_token,
        transformer=None, vae=None)
    for enc in [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]:
        if enc is not None:
            enc.to(gpu_device)

    # Encode negative (empty string) embeddings once and save to cache root
    neg_path = os.path.join(args.cache_dir, "_negative.pt")
    if rank == 0:
        os.makedirs(args.cache_dir, exist_ok=True)
        if not os.path.exists(neg_path):
            print("Encoding negative (empty) prompt...", flush=True)
            with torch.no_grad():
                pe_neg, _, ppe_neg, _ = pipeline.encode_prompt(
                    prompt=[""], prompt_2=None, prompt_3=None,
                    device=gpu_device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
            torch.save({
                "negative_prompt_embeds": pe_neg[0].cpu().float(),
                "negative_pooled_prompt_embeds": ppe_neg[0].cpu().float(),
            }, neg_path)
            print(f"Saved negative embeddings to {neg_path}", flush=True)

    if rank == 0:
        print(f"Encoding on {gpu_device} ({dtype_label}), batch_size={args.batch_size}...", flush=True)

    # Encode only positive prompts (no CFG doubling — halves GPU memory)
    disable_tqdm = rank != 0
    for batch_start in tqdm(range(0, len(my_items), args.batch_size),
                            desc=f"[rank {rank}]", disable=disable_tqdm):
        batch = my_items[batch_start:batch_start + args.batch_size]
        captions = []
        for txt_path, _ in batch:
            with open(txt_path, "r") as f:
                caption = trim_to_77_tokens(f.readline().strip())
            captions.append(caption)

        with torch.no_grad():
            pe, _, ppe, _ = pipeline.encode_prompt(
                prompt=captions, prompt_2=None, prompt_3=None,
                device=gpu_device, num_images_per_prompt=1,
                do_classifier_free_guidance=False)

        for i, (_, cache_path) in enumerate(batch):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                "prompt_embeds": pe[i].cpu().float(),
                "pooled_prompt_embeds": ppe[i].cpu().float(),
            }, cache_path)

    print(f"[rank {rank}] Done. Cached {len(my_items)} prompts.", flush=True)


if __name__ == "__main__":
    main()
