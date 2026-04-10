"""Regenerate w_scale_analysis plot from a checkpoint."""
import os, sys, argparse, torch
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.model.guidance_scale_model import ScalarMLP


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt.get("model_config") or ckpt.get("config", {}).get("guidance_scale_model", {}))
    sd = ckpt.get("model_state_dict") or ckpt.get("guidance_scale_model")
    cfg.setdefault(
        "num_timesteps",
        ckpt.get("config", {}).get("diffusion", {}).get("num_sampling_steps")
        or ckpt.get("config", {}).get("diffusion", {}).get("num_timesteps"),
    )
    model = ScalarMLP(**cfg).to(device)
    model.num_timesteps = cfg.get("num_timesteps")
    model.load_state_dict(sd, strict=True)
    model.eval()
    step = ckpt.get("step", "?")
    return model, cfg, step


def _analysis_kwargs(model, t_batch, device):
    kwargs = {}
    if getattr(model, "interval_embed_dim", 0) > 0:
        num_steps = max(int(getattr(model, "num_timesteps", 20) or 20), 1)
        next_t = torch.clamp(t_batch - (1000.0 / num_steps), min=0.0)
        kwargs["interval"] = (t_batch - next_t) / 1000.0
    if getattr(model, "c_embed_dim", 0) > 0:
        kwargs["c_emb"] = torch.zeros((t_batch.shape[0], model.c_proj.in_features),
                                       device=device, dtype=torch.float32)
    return kwargs


@torch.no_grad()
def eval_model(model, timesteps, lambda_vals, delta_norm, device="cpu"):
    """Evaluate model for a grid of (timestep, lambda) at fixed delta_norm."""
    results = {}
    for lam in lambda_vals:
        ws = []
        for t in timesteps:
            uncond = torch.full((1, 1, 1, 1), delta_norm, device=device)
            text = torch.zeros(1, 1, 1, 1, device=device)
            t_batch = torch.tensor([t], dtype=torch.float32, device=device)
            w = model(t_batch,
                      torch.tensor([lam], dtype=torch.float32, device=device),
                      uncond, text,
                      **_analysis_kwargs(model, t_batch, device))
            ws.append(w.item())
        results[lam] = ws
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Default: results/w_scale_analysis.png")
    parser.add_argument("--lr_label", type=str, default="lr=1e-3")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_REPO_ROOT, ckpt_path)

    model, cfg, step = load_model(ckpt_path)
    dn_norm = cfg.get("delta_embed_normalization", 5.0)

    timesteps = np.linspace(0, 1000, 200)
    lambda_vals = [0.0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    # Three delta norms: small, medium, larger
    delta_norms = [1.0 * dn_norm, 5.1 * dn_norm, 10.2 * dn_norm]
    delta_labels = [
        f"Small delta (norm/{dn_norm:.0f}=1.0)",
        f"Medium delta (norm/{dn_norm:.0f}=5.1)",
        f"Larger delta (norm/{dn_norm:.0f}=10.2)",
    ]

    # Colour map: blue (low lambda) -> red (high lambda)
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(lambda_vals) - 1)) for i in range(len(lambda_vals))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    all_ws = []
    for ax, dn, dl in zip(axes, delta_norms, delta_labels):
        results = eval_model(model, timesteps, lambda_vals, dn)
        for lam, color in zip(lambda_vals, colors):
            label = f"lambda={lam:.2f}" if lam not in (0.0, 1.0) else f"lambda={lam:.1f}"
            if lam == 1.0:
                pass
            ax.plot(timesteps, results[lam], color=color, label=label, linewidth=1.5)
            all_ws.extend(results[lam])
        ax.set_title(dl, fontsize=12)
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Guidance Scale (w)")

    w_min, w_max = min(all_ws), max(all_ws)
    fig.suptitle(
        f"Learned w(t, λ) - ckpt step {step} ({args.lr_label})\n"
        f"w range: {w_min:.2f}-{w_max:.2f}",
        fontsize=13,
    )
    fig.tight_layout()

    out = args.output or os.path.join(_REPO_ROOT, "results", "w_scale_analysis.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
