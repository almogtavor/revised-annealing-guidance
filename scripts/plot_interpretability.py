"""Interpretability plots for the guidance-scale MLP.

All plots are checkpoint-only (no pipeline/inference needed), so they run fast.

Generates 5 plots:
  1. Lambda Steering Curves — w vs λ at representative timesteps
  2. Dynamic Range Band — w range [λ=0, λ=1] per timestep
  3. Delta Sensitivity — w vs delta_norm at fixed λ and several timesteps
  4. Lambda Sensitivity Heatmap — ∂w/∂λ over (t, delta_norm) grid
  5. Guidance Gradient — ∂w/∂t for different lambdas
"""
import os
import sys
import argparse

import torch
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_REPO_ROOT))
from src.model.guidance_scale_model import ScalarMLP


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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
        num_timesteps=cfg.get("num_timesteps") or ckpt.get("config", {}).get("diffusion", {}).get("num_sampling_steps") or ckpt.get("config", {}).get("diffusion", {}).get("num_timesteps"),
        delta_embed_normalization=cfg.get("delta_embed_normalization", 5.0),
        w_bias=cfg.get("w_bias", 1.0),
        w_scale=cfg.get("w_scale", 1.0),
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    step = ckpt.get("step", "?")
    label = ckpt.get("config", {}).get("training", {}).get("label", "")
    return model, cfg, step, label


@torch.no_grad()
def eval_w(model, timesteps, delta_norm, lam, device="cpu"):
    """Evaluate w for arrays of timesteps at fixed delta_norm and lambda.
    Returns 1D array of w values."""
    n = len(timesteps)
    uncond = torch.full((n, 1, 1, 1), delta_norm, device=device, dtype=torch.float32)
    text = torch.zeros(n, 1, 1, 1, device=device, dtype=torch.float32)
    t_batch = torch.tensor(timesteps, device=device, dtype=torch.float32)
    l_batch = torch.full((n,), lam, device=device, dtype=torch.float32)
    w = model(t_batch, l_batch, uncond, text)
    return w.cpu().numpy()


# ---------------------------------------------------------------------------
# Plot 1: Lambda Steering Curves
# ---------------------------------------------------------------------------

def plot_lambda_steering(model, output_path, device="cpu"):
    """w vs λ ∈ [0,1] at representative timesteps and fixed delta_norm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = np.linspace(0, 1, 50)
    delta_norm = 5.0
    timesteps_to_show = [50, 250, 500, 750, 950]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(timesteps_to_show)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for t_val, color in zip(timesteps_to_show, colors):
        ws = []
        for lam in lambdas:
            w = eval_w(model, [t_val], delta_norm, lam, device)
            ws.append(w[0])
        ax.plot(lambdas, ws, color=color, linewidth=2.5, label=f"t={t_val}")

    ax.set_xlabel("λ (lambda)", fontsize=13)
    ax.set_ylabel("Guidance scale w", fontsize=13)
    ax.set_title("Lambda Steering Curves\n(how λ controls w at different timesteps)", fontsize=13)
    ax.legend(fontsize=10, title="Timestep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Dynamic Range Band
# ---------------------------------------------------------------------------

def plot_dynamic_range(model, output_path, device="cpu"):
    """w range [λ=0, λ=1] per timestep — shows where lambda has control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = np.linspace(1, 999, 100)
    delta_norm = 5.0

    w_low = eval_w(model, timesteps, delta_norm, 0.0, device)
    w_high = eval_w(model, timesteps, delta_norm, 1.0, device)
    w_mid = eval_w(model, timesteps, delta_norm, 0.5, device)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(timesteps, w_low, w_high, alpha=0.25, color="#1f77b4",
                     label="w range (λ=0 to λ=1)")
    ax.plot(timesteps, w_mid, color="#1f77b4", linewidth=2, label="λ=0.5")
    ax.plot(timesteps, w_low, color="#1f77b4", linewidth=1, linestyle="--", alpha=0.6, label="λ=0")
    ax.plot(timesteps, w_high, color="#1f77b4", linewidth=1, linestyle="--", alpha=0.6, label="λ=1")

    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Guidance scale w", fontsize=13)
    ax.set_title("Dynamic Range: where does λ matter?\n(wider band = more user control)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1000)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Delta Sensitivity
# ---------------------------------------------------------------------------

def plot_delta_sensitivity(model, output_path, device="cpu"):
    """w vs delta_norm at fixed λ=0.5 and several timesteps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta_norms = np.linspace(0.1, 15, 60)
    lam = 0.5
    timesteps_to_show = [50, 250, 500, 750, 950]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(timesteps_to_show)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for t_val, color in zip(timesteps_to_show, colors):
        ws = []
        for dn in delta_norms:
            w = eval_w(model, [t_val], dn, lam, device)
            ws.append(w[0])
        ax.plot(delta_norms, ws, color=color, linewidth=2.5, label=f"t={t_val}")

    ax.set_xlabel("||δ|| (delta norm)", fontsize=13)
    ax.set_ylabel("Guidance scale w", fontsize=13)
    ax.set_title("Delta Sensitivity (λ=0.5)\n(how noise gap magnitude affects guidance)", fontsize=13)
    ax.legend(fontsize=10, title="Timestep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Lambda Sensitivity Heatmap (∂w/∂λ)
# ---------------------------------------------------------------------------

def plot_lambda_sensitivity_heatmap(model, output_path, device="cpu"):
    """∂w/∂λ over (timestep, delta_norm) grid at λ=0.5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_t, n_d = 60, 50
    timesteps = np.linspace(1, 999, n_t)
    delta_norms = np.linspace(0.5, 12, n_d)
    lam_center = 0.5
    eps = 0.05

    sensitivity = np.zeros((n_d, n_t))
    for di, dn in enumerate(delta_norms):
        w_plus = eval_w(model, timesteps, dn, lam_center + eps, device)
        w_minus = eval_w(model, timesteps, dn, lam_center - eps, device)
        sensitivity[di, :] = (w_plus - w_minus) / (2 * eps)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(sensitivity, aspect="auto", origin="lower",
                    extent=[timesteps[0], timesteps[-1], delta_norms[0], delta_norms[-1]],
                    cmap="RdBu_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("∂w/∂λ", fontsize=12)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("||δ|| (delta norm)", fontsize=13)
    ax.set_title("Lambda Sensitivity: ∂w/∂λ at λ=0.5\n(where does changing λ have the biggest effect?)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Guidance Gradient ∂w/∂t
# ---------------------------------------------------------------------------

def plot_guidance_gradient(model, output_path, device="cpu"):
    """∂w/∂t for different lambdas — smooth vs abrupt guidance transitions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = np.linspace(1, 999, 200)
    delta_norm = 5.0
    lambdas_to_show = [0.0, 0.2, 0.5, 0.8, 1.0]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(lambdas_to_show)))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top: w(t) curves
    ax1 = axes[0]
    for lam, color in zip(lambdas_to_show, colors):
        ws = eval_w(model, timesteps, delta_norm, lam, device)
        ax1.plot(timesteps, ws, color=color, linewidth=2, label=f"λ={lam:.1f}")
    ax1.set_ylabel("w", fontsize=13)
    ax1.set_title("Guidance scale w(t) and its gradient ∂w/∂t", fontsize=13)
    ax1.legend(fontsize=9, ncol=len(lambdas_to_show))
    ax1.grid(True, alpha=0.3)

    # Bottom: ∂w/∂t (finite differences)
    ax2 = axes[1]
    dt = timesteps[1] - timesteps[0]
    for lam, color in zip(lambdas_to_show, colors):
        ws = eval_w(model, timesteps, delta_norm, lam, device)
        dwdt = np.gradient(ws, dt)
        ax2.plot(timesteps, dwdt, color=color, linewidth=2, label=f"λ={lam:.1f}")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Timestep", fontsize=13)
    ax2.set_ylabel("∂w/∂t", fontsize=13)
    ax2.legend(fontsize=9, ncol=len(lambdas_to_show))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP Interpretability Plots")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plots (alongside heatmap/w_scale)")
    parser.add_argument("--lr_label", type=str, default="")
    args = parser.parse_args()

    device = "cpu"
    model, cfg, step, label = load_model(args.checkpoint, device)
    run_label = args.lr_label or label or "unknown"

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print(f"Generating interpretability plots (step {step}, {run_label})...")

    plot_lambda_steering(model, os.path.join(out, "interp_lambda_steering.png"), device)
    plot_dynamic_range(model, os.path.join(out, "interp_dynamic_range.png"), device)
    plot_delta_sensitivity(model, os.path.join(out, "interp_delta_sensitivity.png"), device)
    plot_lambda_sensitivity_heatmap(model, os.path.join(out, "interp_lambda_sensitivity.png"), device)
    plot_guidance_gradient(model, os.path.join(out, "interp_guidance_gradient.png"), device)

    print("Interpretability plots done.")


if __name__ == "__main__":
    main()
