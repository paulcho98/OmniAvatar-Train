"""Plot combined metrics from multiple ODE analysis variants on the same axes.

Usage:
    python scripts/plot_combined_ode_comparison_latentsync.py \
        --output_dir /home/work/.local/ode_analysis/latentsync_1.6/combined
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
    "trajectory": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/perceptual_v2/metrics.csv",
        "label": "50-step trajectory CFG=1.5",
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "trajectory_nocfg": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/trajectory_nocfg/metrics.csv",
        "label": "50-step trajectory noCFG",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.5,
    },
    "euler_cfg15_cfg15": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/euler_cfg15_cfg15/metrics.csv",
        "label": "Euler: CFG",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "euler_nocfg_cfg15": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/euler_nocfg_cfg15/metrics.csv",
        "label": "Euler: noCFG step0",
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "euler_nocfg_nocfg": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/euler_nocfg_nocfg/metrics.csv",
        "label": "Euler: noCFG",
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.5,
    },
}

TRAJ_DIR = "/home/work/.local/ode_full_trajectories/latentsync_1.6"


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

    # Load all variants
    dfs = {}
    for name, cfg in VARIANTS.items():
        if not os.path.exists(cfg["path"]):
            print(f"Skipping {name}: {cfg['path']} not found")
            continue
        df = pd.read_csv(cfg["path"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df[df["step"] >= 0]
        dfs[name] = df.groupby(["step", "metric", "region"])["value"].mean().reset_index()
        print(f"Loaded {name}: {len(df)} rows")

    # ── Reference metrics (mouth only): MSE, SSIM, LPIPS, LMD ──
    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim", "SSIM (mouth)", False),
        ("lpips", "LPIPS (mouth)", False),
        ("lmd", "LMD (mouth)", False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    fig.suptitle("LatentSync 1.6 — ODE Single-Step Prediction: Euler Jump CFG Ablation (mouth region)",
                 fontsize=16, fontweight="bold")

    for col, (metric_name, title, use_log) in enumerate(ref_metrics):
        ax = axes[0, col]

        for name, cfg in VARIANTS.items():
            if name not in dfs:
                continue
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
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

        # Upper face subplot
        ax_uf = axes[1, col]
        for name, cfg in VARIANTS.items():
            if name not in dfs:
                continue
            sub = dfs[name]
            region = "upper_face" if metric_name != "lmd" else "mouth"
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == region)].sort_values("step")
            if len(data) == 0:
                continue
            ax_uf.plot(data["step"], data["value"],
                       color=cfg["color"], linestyle=cfg["linestyle"],
                       linewidth=cfg["linewidth"], label=cfg["label"],
                       marker=".", markersize=2)

        region_label = "upper face" if metric_name != "lmd" else "mouth (delta)"
        ax_uf.set_xlabel("ODE Step")
        if metric_name != "lmd":
            ax_uf.set_ylabel(f"{title.split('(')[0]}({region_label})")
            ax_uf.set_title(f"{title.split('(')[0]}({region_label})")
        else:
            # For LMD, show per-step delta instead of upper face
            ax_uf.set_ylabel("Δ LMD (per step)")
            ax_uf.set_title("Per-Step Δ LMD (mouth)")
            ax_uf.cla()
            for vname, cfg in VARIANTS.items():
                if vname not in dfs:
                    continue
                sub = dfs[vname]
                data = sub[(sub["metric"] == "lmd") & (sub["region"] == "mouth")].sort_values("step")
                if len(data) == 0:
                    continue
                vals = data["value"].values
                delta = np.zeros(len(vals))
                delta[1:] = vals[:-1] - vals[1:]  # decrease = improvement
                ax_uf.plot(data["step"].values, delta,
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.axhline(y=0, color="gray", linewidth=0.5)
            ax_uf.set_xlabel("ODE Step")
            ax_uf.set_ylabel("Δ LMD (improvement)")
            ax_uf.set_title("Per-Step Δ LMD")

        ax_uf.legend(fontsize=8)
        ax_uf.grid(True, alpha=0.3)
        if use_log and metric_name != "lmd":
            ax_uf.set_yscale("log")

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "reference_metrics_combined.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved {path1}")

    # ── No-reference metrics: Sharpness, Sync-D, Sync-C ──
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d", "Sync-D (lower=better)"),
        ("sync_c", "Sync-C (higher=better)"),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("LatentSync 1.6 — No-Reference Metrics: Euler Jump CFG Ablation",
                  fontsize=14, fontweight="bold")

    # Get GT baselines from trajectory variant (average over samples)
    gt_baselines = {}
    if "trajectory" in dfs:
        traj_csv = pd.read_csv(VARIANTS["trajectory"]["path"])
        traj_csv["step"] = pd.to_numeric(traj_csv["step"], errors="coerce")
        gt_rows = traj_csv[traj_csv["step"] == -1]
        gt_agg = gt_rows.groupby(["metric", "region"])["value"].mean()
        for (metric, region), val in gt_agg.items():
            gt_baselines[(metric, region)] = val

    for col, (metric_name, title) in enumerate(noref_metrics):
        ax = axes2[col]

        for name, cfg in VARIANTS.items():
            if name not in dfs:
                continue
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)

        # GT baseline
        gt_key = (metric_name, "mouth")
        if gt_key in gt_baselines:
            gt_val = gt_baselines[gt_key]
            ax.axhline(y=gt_val, color="green", linestyle="--",
                       linewidth=2, label=f"GT ({gt_val:.2f})")

        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "noref_metrics_combined.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
