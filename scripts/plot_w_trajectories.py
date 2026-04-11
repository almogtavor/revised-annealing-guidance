"""Plot w trajectories from sampling meta.json files.

Center: mean w(t) ± std across all prompts, per lambda (+ auto-lambda dashed green, FSG dashed gray).
Left/Right: individual trajectories for a small-delta and large-delta prompt.

Usage: python scripts/plot_w_trajectories.py --results_dir results/final/XXXX_label/label
"""
import os, sys, json, argparse, glob
import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_w_histories(results_dir):
    """Scan all meta.json files, return {method_key: [(prompt_id, [(t, w), ...]), ...]}."""
    data = {}
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "fig_*", "prompt_*", "*", "meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        wh = meta.get("w_history", [])
        if not wh:
            continue
        # Determine method key from directory name
        method_dir = os.path.basename(os.path.dirname(meta_path))
        pid = meta.get("prompt_id", 0)
        data.setdefault(method_dir, []).append((pid, wh))
    return data


def trajectories_to_array(entries):
    """Convert list of (pid, [(t,w),...]) to (timesteps, w_matrix[n_prompts, n_steps])."""
    if not entries:
        return None, None
    ts = np.array([t for t, w in entries[0][1]])
    ws = np.array([[w for t, w in wh] for _, wh in entries])
    return ts, ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = load_w_histories(args.results_dir)
    if not data:
        print(f"No w_history found in {args.results_dir}"); return

    # Group methods
    lambda_keys = sorted([k for k in data if k.startswith("lambda_")],
                         key=lambda k: float(k.split("_")[1]))
    auto_key = "auto_lambda" if "auto_lambda" in data else None
    fsg_keys = sorted([k for k in data if k.startswith("fsg_")])

    # Color map for lambda values
    cmap = plt.cm.coolwarm
    n_lam = max(len(lambda_keys), 1)

    # Find small-delta and large-delta prompts (by mean |w| across all lambdas)
    all_prompt_ids = set()
    prompt_mean_w = {}
    for k in lambda_keys:
        for pid, wh in data[k]:
            all_prompt_ids.add(pid)
            ws = [abs(w) for _, w in wh]
            prompt_mean_w.setdefault(pid, []).extend(ws)
    prompt_avg = {pid: np.mean(vs) for pid, vs in prompt_mean_w.items()}
    if len(prompt_avg) >= 2:
        sorted_pids = sorted(prompt_avg, key=lambda p: prompt_avg[p])
        small_pid, large_pid = sorted_pids[0], sorted_pids[-1]
    else:
        small_pid = large_pid = list(all_prompt_ids)[0] if all_prompt_ids else 0

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    ax_small, ax_mean, ax_large = axes

    def plot_methods(ax, filter_pid=None, show_std=False):
        for i, k in enumerate(lambda_keys):
            color = cmap(i / (n_lam - 1)) if n_lam > 1 else 'blue'
            lam_val = k.split("_")[1]
            entries = [(pid, wh) for pid, wh in data[k] if filter_pid is None or pid == filter_pid]
            ts, ws = trajectories_to_array(entries)
            if ts is None:
                continue
            mean = ws.mean(axis=0)
            ax.plot(ts, mean, color=color, label=f"λ={lam_val}", linewidth=1.5)
            if show_std and ws.shape[0] > 1:
                std = ws.std(axis=0)
                ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

        if auto_key and auto_key in data:
            entries = [(pid, wh) for pid, wh in data[auto_key] if filter_pid is None or pid == filter_pid]
            ts, ws = trajectories_to_array(entries)
            if ts is not None:
                mean = ws.mean(axis=0)
                ax.plot(ts, mean, color='lightgreen', linestyle='--', label='auto-λ', linewidth=2)
                if show_std and ws.shape[0] > 1:
                    std = ws.std(axis=0)
                    ax.fill_between(ts, mean - std, mean + std, color='lightgreen', alpha=0.1)

        for fk in fsg_keys:
            entries = [(pid, wh) for pid, wh in data[fk] if filter_pid is None or pid == filter_pid]
            ts, ws = trajectories_to_array(entries)
            if ts is not None:
                mean = ws.mean(axis=0)
                label = fk.replace("fsg_", "FSG ") if fk == fsg_keys[0] else None
                ax.plot(ts, mean, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                if show_std and ws.shape[0] > 1:
                    std = ws.std(axis=0)
                    ax.fill_between(ts, mean - std, mean + std, color='gray', alpha=0.08)

        ax.set_xlabel("Timestep")
        ax.legend(fontsize=7, loc="best")

    # Left: small delta prompt
    ax_small.set_title(f"Small delta (prompt {small_pid})", fontsize=11)
    ax_small.set_ylabel("Guidance Scale (w)")
    plot_methods(ax_small, filter_pid=small_pid)

    # Center: mean ± std across all prompts
    ax_mean.set_title("Mean ± std (all prompts)", fontsize=11)
    plot_methods(ax_mean, show_std=True)

    # Right: large delta prompt
    ax_large.set_title(f"Large delta (prompt {large_pid})", fontsize=11)
    plot_methods(ax_large, filter_pid=large_pid)

    fig.suptitle(f"Observed w trajectories — {os.path.basename(args.results_dir)}", fontsize=13)
    fig.tight_layout()

    out = args.output or os.path.join(args.results_dir, "w_trajectories.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
