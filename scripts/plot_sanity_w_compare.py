"""Plot mean guidance-scale (w) vs training step for all four sanity runs.

For each checkpoint in a run, loads the model, evaluates w across diffusion
timesteps, and plots the mean w at that training step.

Produces results/sanity_w_compare.png (and .pdf).
"""
import glob
import os
import re
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from src.model.guidance_scale_model import ScalarMLP


# ── run definitions ──────────────────────────────────────────────────────
RUNS = [
    {
        "label": "delta-only CFG++ (lambda=1)",
        "ckpt_dir": "output/sanity_delta_only_v2",
        "color": "#d95f02",
        "ls": "-",
    },
    {
        "label": "eps-only CFG++ (lambda=0)",
        "ckpt_dir": "output/sanity_eps_only",
        "color": "#1b9e77",
        "ls": "-",
    },
    {
        "label": "delta-only CFG (lambda=1)",
        "ckpt_dir": "output/sanity_delta_only_cfg_v2",
        "color": "#d95f02",
        "ls": "--",
    },
    {
        "label": "eps-only CFG (lambda=0)",
        "ckpt_dir": "output/sanity_eps_only_cfg_v2",
        "color": "#1b9e77",
        "ls": "--",
    },
]


def find_all_checkpoints(ckpt_dir, latest_dir_only=False, exclude_dirs=None):
    """Return sorted list of (step, path) for all checkpoints."""
    exclude_dirs = set(exclude_dirs or [])
    pattern_direct = os.path.join(ckpt_dir, "checkpoint_step_*.pt")
    if latest_dir_only:
        subdirs = [d for d in sorted(glob.glob(os.path.join(ckpt_dir, "checkpoints_*")))
                   if os.path.basename(d) not in exclude_dirs]
        matches = glob.glob(os.path.join(subdirs[-1], "checkpoint_step_*.pt")) if subdirs else []
    else:
        matches = glob.glob(pattern_direct) or glob.glob(os.path.join(ckpt_dir, "checkpoints_*", "checkpoint_step_*.pt"))
    if not matches:
        return []
    step_re = re.compile(r"checkpoint_step_(\d+)\.pt$")
    by_step = {}
    for p in matches:
        m = step_re.search(p)
        if m:
            step = int(m.group(1))
            # Keep newest file per step (later checkpoint dir wins)
            if step not in by_step or os.path.getmtime(p) > os.path.getmtime(by_step[step]):
                by_step[step] = p
    return sorted(by_step.items())


def _build_model(cfg):
    return ScalarMLP(
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],
        n_layers=cfg["n_layers"],
        t_embed_dim=cfg.get("t_embed_dim", 4),
        delta_embed_dim=cfg.get("delta_embed_dim", 4),
        lambda_embed_dim=cfg.get("lambda_embed_dim", 4),
        t_embed_normalization=cfg.get("t_embed_normalization", 1000),
        num_timesteps=cfg.get("num_timesteps"),
        delta_embed_normalization=cfg.get("delta_embed_normalization", 5.0),
        w_scale=cfg.get("w_scale", 1.0),
        w_bias=cfg.get("w_bias", 1.0),
    )


@torch.no_grad()
def eval_mean_w(ckpt_path, num_timesteps=50, delta_norm=5.0):
    """Load checkpoint, evaluate w across timesteps, return mean w."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]["guidance_scale_model"]
    model = _build_model(cfg)
    state_key = "guidance_scale_model" if "guidance_scale_model" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    train_cfg = ckpt["config"].get("training", {})
    lam = train_cfg.get("fixed_lambda", 0.5)

    timesteps = torch.linspace(0, 999, num_timesteps)
    dummy = torch.zeros(1, 1, 1, 1)
    dummy[0, 0, 0, 0] = delta_norm
    zero = torch.zeros_like(dummy)

    w_sum = 0.0
    for t in timesteps:
        w = model(t.unsqueeze(0), lam, dummy, zero)
        w_sum += w.item()
    return w_sum / num_timesteps


def main():
    out_prefix = _REPO_ROOT / "results" / "sanity_w_compare"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)

    for run in RUNS:
        ckpt_dir = _REPO_ROOT / run["ckpt_dir"]
        checkpoints = find_all_checkpoints(str(ckpt_dir), latest_dir_only=run.get("latest_dir_only", False), exclude_dirs=run.get("exclude_dirs"))
        if not checkpoints:
            print(f"  SKIP {run['label']}: no checkpoints in {ckpt_dir}")
            continue

        print(f"  {run['label']}: {len(checkpoints)} checkpoints, "
              f"steps {checkpoints[0][0]}–{checkpoints[-1][0]}")

        steps, mean_ws = [], []
        for step, path in checkpoints:
            mean_w = eval_mean_w(path)
            steps.append(step)
            mean_ws.append(mean_w)
            print(f"    step {step}: mean w = {mean_w:.3f}")

        ax.plot(
            steps, mean_ws,
            color=run["color"], linestyle=run["ls"],
            linewidth=2.0, alpha=0.95,
            label=run["label"],
        )
        ax.scatter([steps[-1]], [mean_ws[-1]], color=run["color"], s=30, zorder=3)
        ax.annotate(
            f"{mean_ws[-1]:.2f}", (steps[-1], mean_ws[-1]),
            xytext=(6, 6), textcoords="offset points", fontsize=9, color=run["color"],
        )

    ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.2, alpha=0.8, label="w = 1 (bias)")
    ax.set_title("Mean guidance scale w vs training step  (CFG++ solid, CFG dashed)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean w (across timesteps)")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        p = out_prefix.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
