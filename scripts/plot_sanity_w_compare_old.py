"""Plot guidance-scale trajectories for the delta-only and epsilon-only sanity runs."""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run(path):
    with path.open("r") as f:
        data = json.load(f)
    rows = data["data"]
    return [row[0] for row in rows], [row[2] for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta-table",
        type=str,
        default="wandb/run-20260319_205204-6ntgx5g5/files/media/table/charts/w_vs_timestep_table_5000_8fe1d9e98c688f44951c.table.json",
    )
    parser.add_argument(
        "--eps-table",
        type=str,
        default="wandb/run-20260316_063618-exo61lob/files/media/table/charts/w_vs_timestep_table_5000_06f8d473b91e11f92449.table.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="results/sanity_w_compare",
        help="Writes both <prefix>.png and <prefix>.pdf",
    )
    args = parser.parse_args()

    delta_path = Path(args.delta_table)
    eps_path = Path(args.eps_table)
    if not delta_path.is_absolute():
        delta_path = _REPO_ROOT / delta_path
    if not eps_path.is_absolute():
        eps_path = _REPO_ROOT / eps_path

    out_prefix = Path(args.output_prefix)
    if not out_prefix.is_absolute():
        out_prefix = _REPO_ROOT / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    delta_steps, delta_w = _load_run(delta_path)
    eps_steps, eps_w = _load_run(eps_path)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)

    ax.plot(delta_steps, delta_w, color="#d95f02", linewidth=2.0, alpha=0.95, label="delta-only (lambda=1)")
    ax.plot(eps_steps, eps_w, color="#1b9e77", linewidth=2.0, alpha=0.95, label="epsilon-only (lambda=0)")
    ax.scatter([delta_steps[-1]], [delta_w[-1]], color="#d95f02", s=30, zorder=3)
    ax.scatter([eps_steps[-1]], [eps_w[-1]], color="#1b9e77", s=30, zorder=3)
    ax.annotate(f"{delta_w[-1]:.2f}", (delta_steps[-1], delta_w[-1]), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#d95f02")
    ax.annotate(f"{eps_w[-1]:.2f}", (eps_steps[-1], eps_w[-1]), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#1b9e77")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2, alpha=0.8, label="bias = 1")

    ax.set_title("Guidance scale w during sanity runs")
    ax.set_xlabel("Training step")
    ax.set_ylabel("w")
    ax.set_xlim(0, max(delta_steps[-1], eps_steps[-1]))
    ax.legend(frameon=True)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
