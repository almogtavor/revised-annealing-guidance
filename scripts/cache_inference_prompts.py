"""Cache inference prompts (57 from batch_sample) + eval prompts (1000 COCO) in fp32.

Uses pipeline.encode_prompt for correctness. Requires >=24GB GPU.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.batch_sample_sd3 import PROMPTS_BY_FIGURE
from scripts.eval_metrics import get_coco_prompts, NUM_IMAGES


def cache_prompts(pipeline, device, prompts_dict, cache_path):
    """Encode prompts using pipeline.encode_prompt (no CFG), save shared negative once."""
    if os.path.exists(cache_path):
        existing = torch.load(cache_path, map_location="cpu")
        missing = {p for p in prompts_dict if p not in existing}
        if not missing:
            print(f"  All {len(prompts_dict)} prompts already cached at {cache_path}")
            return
        print(f"  {len(existing)} cached, {len(missing)} to encode")
        cache = existing
    else:
        cache = {}
        missing = set(prompts_dict.keys())

    # Encode negative once
    with torch.no_grad():
        npe, _, nppe, _ = pipeline.encode_prompt(
            prompt=[""], prompt_2=None, prompt_3=None,
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
    npe_f32, nppe_f32 = npe[0].cpu().float(), nppe[0].cpu().float()

    with torch.no_grad():
        for i, p in enumerate(sorted(missing)):
            pe, _, ppe, _ = pipeline.encode_prompt(
                prompt=p, prompt_2=None, prompt_3=None,
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
            cache[p] = (pe.cpu().float(), npe_f32.unsqueeze(0), ppe.cpu().float(), nppe_f32.unsqueeze(0))
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(missing)}")
                torch.save(cache, cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"  Saved {len(cache)} prompts to {cache_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    from diffusers import StableDiffusion3Pipeline
    print("Loading text encoders (fp32)...", flush=True)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float32, token=hf_token, transformer=None, vae=None)
    pipeline.enable_sequential_cpu_offload(gpu_id=0)

    # 1. Inference prompts (57)
    inf_path = os.path.join(repo_root, "prompt_cache", "_inference_prompts.pt")
    all_inf = {p: True for p in {p for prompts in PROMPTS_BY_FIGURE.values() for _, p in prompts}}
    print(f"Caching {len(all_inf)} inference prompts...", flush=True)
    cache_prompts(pipeline, device, all_inf, inf_path)

    # 2. Eval prompts (1000 COCO)
    eval_path = os.path.join(repo_root, "prompt_cache", "_eval_prompts.pt")
    coco_dir = os.path.join(repo_root, "src", "data", "coco")
    coco_prompts = get_coco_prompts(coco_dir)
    all_eval = {cap: True for _, cap in coco_prompts}
    print(f"Caching {len(all_eval)} eval prompts...", flush=True)
    cache_prompts(pipeline, device, all_eval, eval_path)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
