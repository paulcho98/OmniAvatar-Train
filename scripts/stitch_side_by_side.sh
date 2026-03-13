#!/bin/bash
# Stitch GT (left) and generated (right) videos side by side for visual comparison.
# Covers both ablation combos and prior eval steps, reconstruction datasets only.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_BASE="${PROJECT_DIR}/demo_out/v2v_stitched"
MAX_PARALLEL=4

# GT paths
GT_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data"
GT_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data"

get_gt_dir() {
    local ds=$1
    case "$ds" in
        hdtf) echo "$GT_HDTF" ;;
        hallo3) echo "$GT_HALLO3" ;;
    esac
}

stitch_one() {
    local gt_video=$1 gen_video=$2 out_video=$3
    local gt_h
    gt_h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$gt_video")
    # Make height even for codec compatibility
    gt_h=$(( (gt_h / 2) * 2 ))
    mkdir -p "$(dirname "$out_video")"
    ffmpeg -i "$gt_video" -i "$gen_video" \
        -filter_complex "[0:v]scale=-2:${gt_h}[gt];[1:v]scale=-2:${gt_h}[gen];[gt][gen]hstack=inputs=2" \
        -c:v libx264 -preset fast -crf 18 -c:a aac -map 1:a? -shortest -y \
        "$out_video" </dev/null 2>/dev/null
}

stitch_dir() {
    local gen_dir=$1 out_dir=$2 ds=$3
    local gt_base
    gt_base=$(get_gt_dir "$ds")
    local count=0 total=0 running=0

    total=$(ls "$gen_dir"/*.mp4 2>/dev/null | wc -l)
    [ "$total" -eq 0 ] && echo "    [SKIP] No videos in $gen_dir" && return

    for gen_video in "$gen_dir"/*.mp4; do
        local name
        name=$(basename "$gen_video" .mp4)
        local gt_video="${gt_base}/${name}/sub_clip.mp4"
        local out_video="${out_dir}/${name}.mp4"

        if [ ! -f "$gt_video" ]; then
            echo "    [WARN] No GT for ${name}"
            continue
        fi
        if [ -f "$out_video" ]; then
            count=$((count + 1))
            continue
        fi

        stitch_one "$gt_video" "$gen_video" "$out_video" &
        running=$((running + 1))
        count=$((count + 1))

        if [ "$running" -ge "$MAX_PARALLEL" ]; then
            wait -n 2>/dev/null || true
            running=$((running - 1))
        fi
    done
    wait
    echo "    Done: ${count}/${total} videos"
}

# ─── Ablation ─────────────────────────────────────────────────────────────────
echo "=== Ablation stitching ==="
COMBOS=(2500_nocfg 2500_noffo 5500_nocfg 5500_noffo 5500_cfg)
for combo in "${COMBOS[@]}"; do
    for ds in hdtf hallo3; do
        gen_dir="${PROJECT_DIR}/demo_out/v2v_eval_ablation/${combo}/${ds}_composited"
        out_dir="${OUTPUT_BASE}/ablation/${combo}/${ds}"
        if [ ! -d "$gen_dir" ]; then
            echo "  [SKIP] ${combo}/${ds} (no composited dir)"
            continue
        fi
        echo "  ${combo}/${ds}..."
        stitch_dir "$gen_dir" "$out_dir" "$ds"
    done
done

# ─── Prior eval ───────────────────────────────────────────────────────────────
echo ""
echo "=== Eval stitching ==="
STEPS=(step-2500 step-4500 step-5000)
for step in "${STEPS[@]}"; do
    for ds in hdtf hallo3; do
        gen_dir="${PROJECT_DIR}/demo_out/v2v_eval/${step}/${ds}_composited"
        out_dir="${OUTPUT_BASE}/eval/${step}/${ds}"
        if [ ! -d "$gen_dir" ]; then
            echo "  [SKIP] ${step}/${ds} (no composited dir)"
            continue
        fi
        echo "  ${step}/${ds}..."
        stitch_dir "$gen_dir" "$out_dir" "$ds"
    done
done

echo ""
echo "=== Summary ==="
total=$(find "$OUTPUT_BASE" -name "*.mp4" 2>/dev/null | wc -l)
echo "Total stitched videos: $total"
echo "Output: $OUTPUT_BASE"
