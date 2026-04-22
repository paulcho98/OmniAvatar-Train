"""Exp 1 — scheduled CFG vs full CFG vs 2-step Euler.

Tests whether the 2-step noCFG→CFG operating point's win comes from step
count OR from CFG scheduling. Overlays five curves on the same axes:

  • 50-step trajectory CFG=4.5 (audio-only CFG)       — full CFG
  • 50-step trajectory noCFG                          — full noCFG
  • 50-step trajectory scheduled  (noCFG 0-24, CFG 25-49)  — Exp 1
  • 2-step Euler: noCFG → CFG=4.5                     — euler_nocfg_cfg45
  • 2-step Euler: CFG=4.5 → CFG=4.5                   — euler_cfg45_cfg45

If the scheduled-CFG curve tracks the 2-step noCFG→CFG curve, the win is
about CFG scheduling, not step count. If it tracks the full CFG curve,
it's about step count.

Usage:
    python scripts/plot_exp1_schedule_compare.py --output_dir <dir>
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
    "cfg_full": {
        "path": "/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv",
        "label": "50-step CFG=4.5 (full)",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "nocfg_full": {
        "path": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label": "50-step noCFG (full)",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "scheduled": {
        "path": "/home/work/.local/ode_analysis/14B_audio_only_cfg_schedule25/metrics.csv",
        "label": "50-step scheduled (noCFG 0-24, CFG 25-49)",
        "color": "tab:red",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "euler_nocfg_cfg45": {
        "path": "/home/work/.local/ode_analysis/14B_audio_only_cfg/euler_nocfg_cfg45/metrics.csv",
        "label": "2-step Euler: noCFG → CFG=4.5",
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "euler_cfg45_cfg45": {
        "path": "/home/work/.local/ode_analysis/14B_audio_only_cfg/euler_cfg45_cfg45/metrics.csv",
        "label": "2-step Euler: CFG=4.5 → CFG=4.5",
        "color": "tab:purple",
        "linestyle": "-",
        "linewidth": 1.8,
    },
}

TRAJ_DIR = "/home/work/.local/ode_full_trajectories/14B_audio_only_cfg"


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

    dfs, gt_baselines = {}, {}
    for name, cfg in VARIANTS.items():
        if not os.path.exists(cfg["path"]):
            print(f"Skipping {name}: {cfg['path']} missing")
            continue
        df = pd.read_csv(cfg["path"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        gt_agg = df[df["step"] == -1].groupby(["metric", "region"])["value"].mean()
        for (metric, region), val in gt_agg.items():
            gt_baselines.setdefault((metric, region), val)
        df_steps = df[df["step"] >= 0]
        dfs[name] = df_steps.groupby(["step", "metric", "region"])["value"].mean().reset_index()
        print(f"Loaded {name}: {len(df_steps)} step rows")

    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim", "SSIM (mouth)", False),
        ("lpips", "LPIPS (mouth)", False),
        ("lmd", "LMD (mouth)", False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    fig.suptitle("Exp 1: Step Count vs CFG Scheduling",
                 fontsize=16, fontweight="bold")

    for col, (metric, title, use_log) in enumerate(ref_metrics):
        ax = axes[0, col]
        for name, cfg in VARIANTS.items():
            if name not in dfs:
                continue
            sub = dfs[name]
            data = sub[(sub["metric"] == metric) & (sub["region"] == "mouth")].sort_values("step")
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
        tp = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tp)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tp])
        ax2.set_xlabel("Timestep t", fontsize=8)

        ax_uf = axes[1, col]
        if metric != "lmd":
            for name, cfg in VARIANTS.items():
                if name not in dfs:
                    continue
                sub = dfs[name]
                data = sub[(sub["metric"] == metric) & (sub["region"] == "upper_face")].sort_values("step")
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
                if name not in dfs:
                    continue
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
        ax_uf.legend(fontsize=8)
        ax_uf.grid(True, alpha=0.3)
        if use_log and metric != "lmd":
            ax_uf.set_yscale("log")

    plt.tight_layout()
    p1 = os.path.join(args.output_dir, "exp1_schedule_compare_reference.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Saved {p1}")

    # No-reference metrics including SyncNet
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d", "Sync-D (lower=better)"),
        ("sync_c", "Sync-C (higher=better)"),
    ]
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("Exp 1: Step Count vs CFG Scheduling — No-Reference Metrics",
                  fontsize=14, fontweight="bold")
    for col, (metric, title) in enumerate(noref_metrics):
        ax = axes2[col]
        for name, cfg in VARIANTS.items():
            if name not in dfs:
                continue
            sub = dfs[name]
            data = sub[(sub["metric"] == metric) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)
        gt_key = (metric, "mouth")
        if gt_key in gt_baselines:
            ax.axhline(y=gt_baselines[gt_key], color="green", linestyle="--",
                       linewidth=2, label=f"GT ({gt_baselines[gt_key]:.2f})")
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(args.output_dir, "exp1_schedule_compare_noref.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
