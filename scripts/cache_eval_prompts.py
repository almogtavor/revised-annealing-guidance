"""Cache COCO 2017 val prompts (first 1000) in fp32, loading one encoder at a time.

Supports DDP: torchrun --nproc_per_node=8 scripts/cache_eval_prompts.py
Each worker encodes its shard, rank 0 merges and saves.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_metrics import get_coco_prompts, NUM_IMAGES


def encode_prompts_one_encoder(prompts, model_id, hf_token, device):
    """Load each SD3 text encoder separately in fp32, encode prompts, return combined."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    # --- CLIP 1 ---
    print(f"  [gpu {device}] Loading CLIP-1...", flush=True)
    tok1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", token=hf_token)
    enc1 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32, token=hf_token).to(device)
    clip1_embeds, clip1_pooled = [], []
    with torch.no_grad():
        for p in prompts:
            ids = tok1([p], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            out = enc1(**ids, output_hidden_states=True)
            clip1_embeds.append(out.hidden_states[-2].cpu())
            clip1_pooled.append(out.text_embeds.cpu())
    del enc1, tok1; torch.cuda.empty_cache()

    # --- CLIP 2 ---
    print(f"  [gpu {device}] Loading CLIP-2...", flush=True)
    tok2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", token=hf_token)
    enc2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=torch.float32, token=hf_token).to(device)
    clip2_embeds, clip2_pooled = [], []
    with torch.no_grad():
        for p in prompts:
            ids = tok2([p], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            out = enc2(**ids, output_hidden_states=True)
            clip2_embeds.append(out.hidden_states[-2].cpu())
            clip2_pooled.append(out.text_embeds.cpu())
    del enc2, tok2; torch.cuda.empty_cache()

    # --- T5 ---
    print(f"  [gpu {device}] Loading T5-XXL...", flush=True)
    tok3 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", token=hf_token)
    enc3 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", torch_dtype=torch.float32, token=hf_token).to(device)
    t5_embeds = []
    with torch.no_grad():
        for p in prompts:
            ids = tok3([p], padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)
            out = enc3(**ids)
            t5_embeds.append(out.last_hidden_state.cpu())
    del enc3, tok3; torch.cuda.empty_cache()

    # --- Combine ---
    results = {}
    for i, p in enumerate(prompts):
        clip_embed = torch.cat([clip1_embeds[i], clip2_embeds[i]], dim=-1)
        t5_embed = t5_embeds[i]
        clip_embed_padded = torch.nn.functional.pad(clip_embed, (0, t5_embed.shape[-1] - clip_embed.shape[-1]))
        pe = torch.cat([clip_embed_padded, t5_embed], dim=-2)
        ppe = torch.cat([clip1_pooled[i], clip2_pooled[i]], dim=-1)
        results[p] = pe.float(), ppe.float()
    return results


def main():
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cache_path = os.path.join(repo_root, "prompt_cache", "_eval_prompts.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}

    coco_dir = os.path.join(repo_root, "src", "data", "coco")
    all_prompts = get_coco_prompts(coco_dir)
    to_encode = [(img_id, cap) for img_id, cap in all_prompts if cap not in cache]

    # Shard across workers
    my_items = to_encode[rank::world_size]
    if rank == 0:
        print(f"Total: {len(all_prompts)}, already cached: {len(cache)}, "
              f"to encode: {len(to_encode)}, this worker: {len(my_items)}", flush=True)

    if not my_items:
        if rank == 0 and not to_encode:
            print("All prompts cached.", flush=True)
        return

    my_captions = [cap for _, cap in my_items]
    pos = encode_prompts_one_encoder(my_captions, model_id, hf_token, device)

    # Encode negative once (only rank 0 needs to, but all do for simplicity)
    neg = encode_prompts_one_encoder([""], model_id, hf_token, device)
    npe, nppe = neg[""]

    # Gather results to rank 0
    my_cache = {}
    for cap in my_captions:
        pe, ppe = pos[cap]
        my_cache[cap] = (pe, npe, ppe, nppe)

    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("gloo")
        all_caches = [None] * world_size
        dist.all_gather_object(all_caches, my_cache)
        if rank == 0:
            for c in all_caches:
                cache.update(c)
    else:
        cache.update(my_cache)

    if rank == 0:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
        print(f"Saved {len(cache)} prompts to {cache_path}", flush=True)


if __name__ == "__main__":
    main()
