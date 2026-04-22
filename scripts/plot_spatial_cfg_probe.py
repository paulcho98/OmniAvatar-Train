"""Plot Exp 2 spatial CFG probe — line plots for each protocol.

Reads spatial_cfg_probe.csv from each protocol dir and produces:

  1. Raw |Δ| (mouth / upper_face / full) vs step — one panel per protocol.
  2. Relative-normalized |Δ| vs step — ibid.
  3. Mouth/upper_face ratio vs step — one line per protocol (the central question).
  4. CFG-diff minus noise-floor (signal above control) vs step — if noise_floor present.

Usage:
    python scripts/plot_spatial_cfg_probe.py \\
        --probe_dir /home/work/.local/ode_analysis/spatial_cfg_probe \\
        --output_dir /home/work/.local/ode_analysis/spatial_cfg_probe/plots
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTOCOL_ORDER = ["fresh_noise", "trajectory_cfg", "trajectory_nocfg"]
PROTOCOL_COLOR = {
    "fresh_noise":       "tab:blue",
    "trajectory_cfg":    "tab:orange",
    "trajectory_nocfg":  "tab:red",
}
REGION_LINESTYLE = {
    "mouth":      "-",
    "upper_face": "--",
    "full":       ":",
}


def load_probe(probe_dir):
    dfs = {}
    for proto in PROTOCOL_ORDER:
        csv = os.path.join(probe_dir, proto, "spatial_cfg_probe.csv")
        if not os.path.exists(csv) or os.path.getsize(csv) == 0:
            print(f"Missing/empty {csv}")
            continue
        df = pd.read_csv(csv)
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        agg = df.groupby(["step", "t", "metric", "region", "protocol"])["value"].mean().reset_index()
        dfs[proto] = agg
        print(f"Loaded {proto}: {len(df)} rows")
    return dfs


def plot_raw_vs_step(dfs, out_dir, metric_key="cfg_diff_raw", title_prefix="Raw |Δ|"):
    n = len(dfs)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (proto, agg) in zip(axes, dfs.items()):
        sub = agg[agg["metric"] == metric_key]
        for region in ["mouth", "upper_face", "full"]:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=PROTOCOL_COLOR[proto],
                    linestyle=REGION_LINESTYLE[region],
                    linewidth=2, label=region, marker=".", markersize=3)
        ax.set_title(f"{proto}")
        ax.set_xlabel("ODE step (0=noisy, 49=clean)")
        ax.set_ylabel(f"{title_prefix}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"{title_prefix} between CFG=4.5 and CFG=1.0 teacher predictions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"spatial_cfg_{metric_key}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_ratio(dfs, out_dir, metric_key="cfg_diff_raw"):
    """mouth / upper_face ratio vs step, one line per protocol."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for proto, agg in dfs.items():
        sub = agg[agg["metric"] == metric_key]
        mouth = sub[sub["region"] == "mouth"].set_index("step")["value"]
        upper = sub[sub["region"] == "upper_face"].set_index("step")["value"]
        common = mouth.index.intersection(upper.index)
        if len(common) == 0:
            continue
        ratio = mouth.loc[common] / upper.loc[common].replace(0, np.nan)
        ax.plot(common, ratio.values, color=PROTOCOL_COLOR[proto],
                linewidth=2, label=proto, marker=".", markersize=3)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, label="ratio=1")
    ax.set_xlabel("ODE step (0=noisy, 49=clean)")
    ax.set_ylabel("mouth_diff / upper_face_diff")
    ax.set_title(f"Spatial concentration of CFG effect: mouth/upper ratio ({metric_key})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"spatial_cfg_mouth_upper_ratio_{metric_key}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_noise_floor_corrected(dfs, out_dir):
    """|cfg_diff_raw| − |noise_floor|  vs step, per region, per protocol."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(7 * len(dfs), 5), sharey=True)
    if len(dfs) == 1:
        axes = [axes]
    any_plotted = False
    for ax, (proto, agg) in zip(axes, dfs.items()):
        cfg = agg[agg["metric"] == "cfg_diff_raw"]
        nf = agg[agg["metric"] == "noise_floor"]
        if len(nf) == 0:
            ax.text(0.5, 0.5, "no noise_floor",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{proto}")
            continue
        for region in ["mouth", "upper_face", "full"]:
            c = cfg[cfg["region"] == region].set_index("step")["value"]
            n = nf[nf["region"] == region].set_index("step")["value"]
            common = c.index.intersection(n.index)
            if len(common) == 0:
                continue
            corrected = c.loc[common] - n.loc[common]
            ax.plot(common, corrected.values, color=PROTOCOL_COLOR[proto],
                    linestyle=REGION_LINESTYLE[region],
                    linewidth=2, label=region, marker=".", markersize=3)
            any_plotted = True
        ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("ODE step")
        ax.set_ylabel("|cfg_diff_raw| − |noise_floor|")
        ax.set_title(f"{proto}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("CFG effect above noise-floor baseline (x_t+δ control)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if any_plotted:
        path = os.path.join(out_dir, "spatial_cfg_noise_floor_corrected.png")
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir", type=str,
                        default="/home/work/.local/ode_analysis/spatial_cfg_probe")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dfs = load_probe(args.probe_dir)
    if not dfs:
        print("No probe data found. Exiting.")
        return

    plot_raw_vs_step(dfs, args.output_dir, metric_key="cfg_diff_raw",
                     title_prefix="Raw |Δ|")
    plot_raw_vs_step(dfs, args.output_dir, metric_key="cfg_diff_relative",
                     title_prefix="Relative |Δ|")
    plot_ratio(dfs, args.output_dir, metric_key="cfg_diff_raw")
    plot_ratio(dfs, args.output_dir, metric_key="cfg_diff_relative")
    plot_noise_floor_corrected(dfs, args.output_dir)


if __name__ == "__main__":
    main()
