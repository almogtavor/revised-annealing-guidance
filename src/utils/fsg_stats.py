"""FSG inference statistics accumulator and plotter.

A lightweight global recorder that the pipeline appends to during FSG inner loops,
and a plot function that visualizes convergence behavior across all sampled images.

Usage:
    from src.utils.fsg_stats import record, reset, plot
    # In pipeline FSG loop:
    record(timestep=t.item(), iter_idx=k, dz=..., w=..., delta_norm=...)
    # After sampling:
    plot('output_dir/fsg_convergence.png')
"""
import os
from collections import defaultdict


# Each entry: dict with timestep, iter_idx, dz, w, delta_norm
_records = []


def reset():
    global _records
    _records = []


def record(timestep, iter_idx, dz, w, delta_norm):
    """Append one FSG iteration record. Call from pipeline inner loop."""
    _records.append({
        'timestep': float(timestep),
        'iter_idx': int(iter_idx),
        'dz': float(dz),
        'w': float(w),
        'delta_norm': float(delta_norm),
    })


def record_iteration(t, iter_idx, z_t, z_t_prev, vt, vu, w):
    """Compute dz / delta_norm / w stats for one FSG inner iteration, record them,
    and return the new ``z_t_prev`` (a clone of the current z_t) for the next iteration."""
    dz = (z_t - z_t_prev).reshape(z_t.shape[0], -1).norm(dim=1).mean().item()
    delta_norm = (vt.float() - vu.float()).reshape(vu.shape[0], -1).norm(dim=1).mean().item()
    w_val = w.detach().mean().item() if hasattr(w, 'detach') else float(w)
    record(timestep=t.float().item(), iter_idx=iter_idx,
           dz=dz, w=w_val, delta_norm=delta_norm)
    return z_t.clone()


def num_records():
    return len(_records)


def plot(output_path):
    """Generate FSG convergence plot from accumulated records.

    Shows:
      - Top: avg ||z_t^(k+1) - z_t^(k)|| vs iteration index, bucketed by timestep
      - Bottom: avg w vs iteration index, bucketed by timestep
    """
    if not _records:
        print("[fsg_stats] No records to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Bucket records by timestep range (early/mid/late denoising)
    def bucket(t):
        if t > 666:
            return "early (t>666)"
        if t > 333:
            return "mid (333<t≤666)"
        return "late (t≤333)"

    buckets = defaultdict(lambda: defaultdict(list))  # bucket -> iter_idx -> list of records
    for r in _records:
        buckets[bucket(r['timestep'])][r['iter_idx']].append(r)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = {"early (t>666)": "#1f77b4", "mid (333<t≤666)": "#ff7f0e", "late (t≤333)": "#2ca02c"}
    bucket_order = ["early (t>666)", "mid (333<t≤666)", "late (t≤333)"]

    # Top: dz convergence (skip iter 0 since dz is 0 there)
    ax = axes[0]
    for b in bucket_order:
        if b not in buckets:
            continue
        iters = sorted(buckets[b].keys())
        means = [np.mean([r['dz'] for r in buckets[b][k]]) for k in iters]
        stds = [np.std([r['dz'] for r in buckets[b][k]]) for k in iters]
        ax.errorbar(iters, means, yerr=stds, color=colors[b], linewidth=2,
                    marker='o', label=f"{b} (n={sum(len(buckets[b][k]) for k in iters)})",
                    capsize=4)
    ax.set_ylabel(r"$\|z_t^{(k+1)} - z_t^{(k)}\|$", fontsize=13)
    ax.set_title("FSG fixed-point convergence\n(does z_t stabilize across inner iterations?)",
                 fontsize=13)
    ax.legend(fontsize=10, title="Timestep bucket")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Bottom: w stability
    ax2 = axes[1]
    for b in bucket_order:
        if b not in buckets:
            continue
        iters = sorted(buckets[b].keys())
        means = [np.mean([r['w'] for r in buckets[b][k]]) for k in iters]
        stds = [np.std([r['w'] for r in buckets[b][k]]) for k in iters]
        ax2.errorbar(iters, means, yerr=stds, color=colors[b], linewidth=2,
                     marker='o', label=b, capsize=4)
    ax2.set_xlabel("FSG iteration index k", fontsize=13)
    ax2.set_ylabel("Predicted guidance scale w", fontsize=13)
    ax2.set_title("FSG: does w stabilize across iterations?", fontsize=13)
    ax2.legend(fontsize=10, title="Timestep bucket")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  FSG convergence plot saved: {output_path} ({len(_records)} records)")
