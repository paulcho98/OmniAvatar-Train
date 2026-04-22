"""Plot Exp 2 spatial CFG heatmaps — representative-timestep Δ-map figure.

For each protocol, for a set of representative timesteps (t=0.99, 0.77,
0.56, 0.28, 0.09), average the saved diff_map npy arrays across samples
and render as a 2D heatmap. Produces a 3×5 figure (protocols × timesteps).

Usage:
    python scripts/plot_spatial_cfg_heatmaps.py \\
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


PROTOCOL_ORDER = ["fresh_noise", "trajectory_cfg", "trajectory_nocfg"]
REPRESENTATIVE_TS = [0.99, 0.77, 0.56, 0.28, 0.09]  # rough targets


def find_closest_steps(t_list, targets):
    """Return indices in t_list closest to each target t."""
    t_arr = np.asarray(t_list)
    return [int(np.argmin(np.abs(t_arr - tgt))) for tgt in targets]


def load_avg_map(probe_dir, proto, step_idx):
    """Load and average diff_map.npy across all sample subdirs for a given step."""
    maps_root = os.path.join(probe_dir, proto, "maps")
    if not os.path.isdir(maps_root):
        return None
    all_maps = []
    for sample in sorted(os.listdir(maps_root)):
        path = os.path.join(maps_root, sample, f"step_{step_idx:03d}_diff.npy")
        if os.path.exists(path):
            all_maps.append(np.load(path))
    if not all_maps:
        return None
    return np.stack(all_maps).mean(axis=0)  # [H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir", type=str,
                        default="/home/work/.local/ode_analysis/spatial_cfg_probe")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_path", type=str,
                        default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Pick step indices from whichever protocol has schedule.json first.
    t_list = None
    for proto in PROTOCOL_ORDER:
        sched = os.path.join(args.probe_dir, proto, "schedule.json")
        if os.path.exists(sched):
            with open(sched) as f:
                sch = json.load(f)
            t_list = sch["t_list"]
            break
    if t_list is None:
        print("No schedule.json found.")
        return

    step_indices = find_closest_steps(t_list, REPRESENTATIVE_TS)
    actual_ts = [t_list[s] for s in step_indices]

    # Load mask for overlay outline (optional)
    try:
        from PIL import Image
        mask_img = np.array(Image.open(args.mask_path))[:, :, 0]
        import scipy.ndimage as ndi  # optional — used only for contour
        mask_sm = ndi.zoom(mask_img.astype(np.float32),
                           (512 / mask_img.shape[0], 512 / mask_img.shape[1]))
        mask_binary = (mask_sm > 127).astype(np.float32)
    except Exception:
        mask_binary = None

    ncols = len(step_indices)
    nrows = len(PROTOCOL_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = axes[None, :]

    # Compute global vmax per protocol (not shared across protocols — raw scales differ)
    for row, proto in enumerate(PROTOCOL_ORDER):
        proto_maps = []
        for s in step_indices:
            m = load_avg_map(args.probe_dir, proto, s)
            proto_maps.append(m)
        finite_maps = [m for m in proto_maps if m is not None]
        vmax = max(m.max() for m in finite_maps) if finite_maps else 1.0

        for col, (m, s, t) in enumerate(zip(proto_maps, step_indices, actual_ts)):
            ax = axes[row, col]
            if m is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
                if mask_binary is not None:
                    ax.contour(mask_binary, levels=[0.5], colors="cyan", linewidths=0.6)
            if row == 0:
                ax.set_title(f"step {s} (t≈{t:.2f})", fontsize=10)
            if col == 0:
                ax.set_ylabel(proto, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Spatial |Δ| between CFG=4.5 and CFG=1.0 teacher predictions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "spatial_cfg_heatmaps.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
