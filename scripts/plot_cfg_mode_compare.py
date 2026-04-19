"""Plot 3-way comparison: original CFG vs audio-only CFG vs noCFG trajectory.

Usage:
    python scripts/plot_cfg_mode_compare.py \
        --output_dir /home/work/.local/ode_analysis/14B_cfg_mode_compare
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = {
    "original_cfg": {
        "path": "/home/work/.local/ode_analysis/14B/perceptual_v2/metrics.csv",
        "label": "Original CFG (text+audio)",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "audio_only_cfg": {
        "path": "/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv",
        "label": "Audio-only CFG",
        "color": "tab:red",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "nocfg": {
        "path": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label": "noCFG",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.0,
    },
}

TRAJ_DIR = "/home/work/.local/ode_full_trajectories/14B"


def load_schedule():
    samples = sorted([
        d for d in os.listdir(TRAJ_DIR)
        if os.path.isdir(os.path.join(TRAJ_DIR, d))
        and os.path.isfile(os.path.join(TRAJ_DIR, d, "ode_schedule.json"))
    ])
    with open(os.path.join(TRAJ_DIR, samples[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    return schedule["t_list"], schedule["num_steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t_list, num_steps = load_schedule()
    t_values = np.array(t_list[:num_steps])

    dfs = {}
    gt_baselines = {}
    for name, cfg in VARIANTS.items():
        df = pd.read_csv(cfg["path"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        gt_agg = df[df["step"] == -1].groupby(["metric", "region"])["value"].mean()
        for (metric, region), val in gt_agg.items():
            gt_baselines.setdefault((metric, region), val)
        df_steps = df[df["step"] >= 0]
        dfs[name] = df_steps.groupby(["step", "metric", "region"])["value"].mean().reset_index()
        print(f"Loaded {name}: {len(df_steps)} step rows")

    # Reference metrics: 2x4 layout identical to plot_combined_ode_comparison.py
    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim", "SSIM (mouth)", False),
        ("lpips", "LPIPS (mouth)", False),
        ("lmd", "LMD (mouth)", False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    fig.suptitle("50-Step Trajectory — CFG Mode Comparison",
                 fontsize=16, fontweight="bold")

    for col, (metric_name, title, use_log) in enumerate(ref_metrics):
        ax = axes[0, col]
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

        ax_uf = axes[1, col]
        if metric_name != "lmd":
            for name, cfg in VARIANTS.items():
                sub = dfs[name]
                data = sub[(sub["metric"] == metric_name) & (sub["region"] == "upper_face")].sort_values("step")
                if len(data) == 0:
                    continue
                ax_uf.plot(data["step"], data["value"],
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.set_ylabel(f"{title.split('(')[0]}(Upper Face)")
            ax_uf.set_title(f"{title.split('(')[0]}(Upper Face)")
        else:
            for name, cfg in VARIANTS.items():
                sub = dfs[name]
                data = sub[(sub["metric"] == "lmd") & (sub["region"] == "mouth")].sort_values("step")
                if len(data) == 0:
                    continue
                vals = data["value"].values
                delta = np.zeros(len(vals))
                delta[1:] = vals[:-1] - vals[1:]
                ax_uf.plot(data["step"].values, delta,
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.axhline(y=0, color="gray", linewidth=0.5)
            ax_uf.set_ylabel("Δ LMD (improvement)")
            ax_uf.set_title("Per-Step Δ LMD")

        ax_uf.set_xlabel("ODE Step")
        ax_uf.legend(fontsize=9)
        ax_uf.grid(True, alpha=0.3)
        if use_log and metric_name != "lmd":
            ax_uf.set_yscale("log")

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "cfg_mode_compare_reference.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved {path1}")

    # No-reference metrics
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d", "Sync-D (lower=better)"),
        ("sync_c", "Sync-C (higher=better)"),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("50-Step Trajectory — CFG Mode Comparison (No-Reference)",
                  fontsize=14, fontweight="bold")

    for col, (metric_name, title) in enumerate(noref_metrics):
        ax = axes2[col]
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)

        gt_key = (metric_name, "mouth")
        if gt_key in gt_baselines:
            gt_val = gt_baselines[gt_key]
            ax.axhline(y=gt_val, color="green", linestyle="--",
                       linewidth=2, label=f"GT ({gt_val:.2f})")

        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "cfg_mode_compare_noref.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
