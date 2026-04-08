"""Generate heatmap of w(t, ||delta_t||) for each lambda value (cf. paper Fig. 4)."""
import os, sys, argparse, torch
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.model.guidance_scale_model import ScalarMLP


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("guidance_scale_model", {})
    sd = ckpt.get("model_state_dict") or ckpt.get("guidance_scale_model")
    model = ScalarMLP(
        hidden_size=cfg.get("hidden_size", 128),
        output_size=cfg.get("output_size", 1),
        n_layers=cfg.get("n_layers", 2),
        t_embed_dim=cfg.get("t_embed_dim", 4),
        delta_embed_dim=cfg.get("delta_embed_dim", 4),
        lambda_embed_dim=cfg.get("lambda_embed_dim", 4),
        t_embed_normalization=cfg.get("t_embed_normalization", 1e3),
        num_timesteps=cfg.get("num_timesteps"),
        delta_embed_normalization=cfg.get("delta_embed_normalization", 5.0),
        w_bias=cfg.get("w_bias", 1.0),
        w_scale=cfg.get("w_scale", 1.0),
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    step = ckpt.get("step", "?")
    return model, cfg, step


@torch.no_grad()
def eval_heatmap(model, timesteps, delta_norms, lam, device="cpu"):
    """Evaluate w for a grid of (timestep, delta_norm) at a fixed lambda.

    Returns: 2D array of shape (len(delta_norms), len(timesteps)).
    """
    n_t = len(timesteps)
    n_d = len(delta_norms)
    w_grid = np.zeros((n_d, n_t))

    for di, dn in enumerate(delta_norms):
        # Build a batch of size n_t: one entry per timestep, same delta_norm
        # We create dummy 1x1x1x1 predictions whose ||uncond - text|| = dn
        uncond = torch.full((n_t, 1, 1, 1), dn, device=device, dtype=torch.float32)
        text = torch.zeros(n_t, 1, 1, 1, device=device, dtype=torch.float32)
        t_batch = torch.tensor(timesteps, device=device, dtype=torch.float32)
        l_batch = torch.full((n_t,), lam, device=device, dtype=torch.float32)

        w = model(t_batch, l_batch, uncond, text)  # (n_t,)
        w_grid[di] = w.cpu().numpy()

    return w_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Default: results/w_heatmap.png")
    parser.add_argument("--lr_label", type=str, default="")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.4, 0.6, 0.8, 1.0])
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_REPO_ROOT, ckpt_path)

    model, cfg, step = load_model(ckpt_path)
    dn_norm = cfg.get("delta_embed_normalization", 5.0)

    # Grid axes
    timesteps = np.linspace(0, 1000, 200)
    delta_norms = np.linspace(0, 7 * dn_norm, 100)

    lambda_vals = args.lambdas
    n_lam = len(lambda_vals)

    fig, axes = plt.subplots(1, n_lam, figsize=(4.2 * n_lam, 4), sharey=True)
    if n_lam == 1:
        axes = [axes]

    # Compute all heatmaps first to get global color range
    heatmaps = []
    for lam in lambda_vals:
        hm = eval_heatmap(model, timesteps, delta_norms, lam)
        heatmaps.append(hm)

    vmin = min(hm.min() for hm in heatmaps)
    vmax = max(hm.max() for hm in heatmaps)

    for ax, lam, hm in zip(axes, lambda_vals, heatmaps):
        im = ax.imshow(
            hm,
            aspect="auto",
            origin="lower",
            extent=[timesteps[0], timesteps[-1], delta_norms[0] / dn_norm, delta_norms[-1] / dn_norm],
            vmin=vmin, vmax=vmax,
            cmap="viridis",
        )
        if lam == 0.0:
            ax.set_title(f"λ = 0 (vanilla)", fontsize=12)
        elif lam == 1.0:
            ax.set_title(f"λ = {lam:.1f}", fontsize=12)
        else:
            ax.set_title(f"λ = {lam}", fontsize=12)
        ax.set_xlabel("Timestep")

    axes[0].set_ylabel(r"$\|\|\delta_t\|\|$")

    fig.tight_layout()

    # Shared colorbar (added after tight_layout so it doesn't get overlapped)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.04)
    cbar.set_label("Guidance scale (w)", fontsize=11)

    label = f" ({args.lr_label})" if args.lr_label else ""
    fig.suptitle(
        f"Learned w(t, ||δ_t||, λ) - step {step}{label}",
        fontsize=13, y=1.02,
    )

    out = args.output or os.path.join(_REPO_ROOT, "results", "w_heatmap.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
