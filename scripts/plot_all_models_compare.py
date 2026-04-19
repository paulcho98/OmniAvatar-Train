"""Cross-model overlay: OmniAvatar original CFG vs audio-only CFG vs LatentSync 1.6.

Only the 50-step trajectory curves are overlaid (CFG and noCFG variants per model).
Step index on x-axis. Each model's own scheduler t is labelled on a twin axis.

Usage:
    python scripts/plot_all_models_compare.py --output_dir /home/work/.local/ode_analysis/all_models_combined
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {
    "omni_original": {
        "csv_cfg":   "/home/work/.local/ode_analysis/14B/perceptual_v2/metrics.csv",
        "csv_nocfg": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label_cfg":   "OmniAvatar CFG=4.5 (text+audio)",
        "label_nocfg": "OmniAvatar noCFG",
        "color": "tab:blue",
    },
    "omni_audio_only": {
        "csv_cfg":   "/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv",
        "csv_nocfg": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label_cfg":   "OmniAvatar CFG=4.5 (audio-only)",
        "label_nocfg": None,  # same as original, avoid duplicate legend
        "color": "tab:orange",
    },
    "latentsync": {
        "csv_cfg":   "/home/work/.local/ode_analysis/latentsync_1.6/perceptual_v2/metrics.csv",
        "csv_nocfg": "/home/work/.local/ode_analysis/latentsync_1.6/trajectory_nocfg/metrics.csv",
        "label_cfg":   "LatentSync 1.6 CFG=1.5",
        "label_nocfg": "LatentSync 1.6 noCFG",
        "color": "tab:green",
    },
}


def load_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"] >= 0]
    return df.groupby(["step", "metric", "region"])["value"].mean().reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loaded = {}
    for key, m in MODELS.items():
        loaded[key] = {
            "cfg":   load_metrics(m["csv_cfg"])   if os.path.exists(m["csv_cfg"])   else None,
            "nocfg": load_metrics(m["csv_nocfg"]) if os.path.exists(m["csv_nocfg"]) else None,
        }
        print(f"{key}: cfg={loaded[key]['cfg'] is not None}, nocfg={loaded[key]['nocfg'] is not None}")

    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim",      "SSIM (mouth)",      False),
        ("lpips",     "LPIPS (mouth)",     False),
        ("lmd",       "LMD (mouth)",       False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    fig.suptitle("Cross-Model 50-Step Trajectory Comparison (mouth region)",
                 fontsize=16, fontweight="bold")

    for col, (metric, title, use_log) in enumerate(ref_metrics):
        ax = axes[col]
        for key, m in MODELS.items():
            for variant, ls, lw, label_key in [
                ("cfg",   "-",  2.2, "label_cfg"),
                ("nocfg", "--", 1.8, "label_nocfg"),
            ]:
                df = loaded[key][variant]
                if df is None or m[label_key] is None:
                    continue
                sub = df[(df["metric"] == metric) & (df["region"] == "mouth")].sort_values("step")
                if len(sub) == 0:
                    continue
                ax.plot(sub["step"], sub["value"],
                        color=m["color"], linestyle=ls, linewidth=lw,
                        label=m[label_key], marker=".", markersize=2)

        ax.set_xlabel("ODE Step Index (0 = most noisy → 49 = clean)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(args.output_dir, "all_models_reference.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # No-reference metrics
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d",    "Sync-D (lower=better)"),
        ("sync_c",    "Sync-C (higher=better)"),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("Cross-Model No-Reference Metrics", fontsize=14, fontweight="bold")

    for col, (metric, title) in enumerate(noref_metrics):
        ax = axes2[col]
        for key, m in MODELS.items():
            for variant, ls, lw, label_key in [
                ("cfg",   "-",  2.2, "label_cfg"),
                ("nocfg", "--", 1.8, "label_nocfg"),
            ]:
                df = loaded[key][variant]
                if df is None or m[label_key] is None:
                    continue
                sub = df[(df["metric"] == metric) & (df["region"] == "mouth")].sort_values("step")
                if len(sub) == 0:
                    continue
                ax.plot(sub["step"], sub["value"],
                        color=m["color"], linestyle=ls, linewidth=lw,
                        label=m[label_key], marker=".", markersize=2)

        ax.set_xlabel("ODE Step Index")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(args.output_dir, "all_models_noref.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
