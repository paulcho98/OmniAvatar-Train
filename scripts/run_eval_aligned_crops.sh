#!/bin/bash
# Run eval_aligned_crops.py (GT-bbox-aligned SSIM/FID/FVD) for CFG ablation + LatentSync.
# 10 runs across 4 GPUs in batches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/work/.local/miniconda3/envs/latentsync-metrics/bin/python"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/eval_aligned_crops.py"
OUTPUT_BASE="demo_out/v2v_eval_aligned_crops"

cd "$PROJECT_DIR"

# Originals
ORIG_HDTF="demo_out/v2v_eval_cfg_ablation/originals/hdtf"
ORIG_HALLO3="demo_out/v2v_eval_cfg_ablation/originals/hallo3"

# I3D model for FVD
I3D="/home/work/.local/latentsync-metrics/checkpoints/auxiliary/i3d_torchscript.pt"

run_eval() {
    local gpu=$1 label=$2 real_dir=$3 fake_dir=$4
    local out_dir="${OUTPUT_BASE}/${label}"

    if [ -f "$out_dir/metrics_aligned.log" ]; then
        echo "  [GPU $gpu] [SKIP] $label (already done)"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "  [GPU $gpu] $label..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$EVAL_SCRIPT" \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --output_dir "$out_dir" \
        --device cuda:0 \
        --i3d_path "$I3D" \
        > "$out_dir/eval.log" 2>&1
    echo "  [GPU $gpu] [DONE] $label"
}

echo "=== Aligned Crop Evaluation ==="
mkdir -p "$OUTPUT_BASE"

# ─── Batch 1: maskall cfg1.0+3.0 × hdtf+hallo3 (4 GPUs) ────────────────────
echo ""
echo "=== Batch 1: maskall ==="
run_eval 0 "maskall/cfg1.0/hdtf"   "$ORIG_HDTF"   "demo_out/v2v_eval_cfg_ablation/maskall/cfg1.0/step-5500/hdtf_composited"   &
run_eval 1 "maskall/cfg1.0/hallo3" "$ORIG_HALLO3"  "demo_out/v2v_eval_cfg_ablation/maskall/cfg1.0/step-5500/hallo3_composited" &
run_eval 2 "maskall/cfg3.0/hdtf"   "$ORIG_HDTF"   "demo_out/v2v_eval_cfg_ablation/maskall/cfg3.0/step-5500/hdtf_composited"   &
run_eval 3 "maskall/cfg3.0/hallo3" "$ORIG_HALLO3"  "demo_out/v2v_eval_cfg_ablation/maskall/cfg3.0/step-5500/hallo3_composited" &
wait
echo "  Batch 1 complete."

# ─── Batch 2: refseq cfg1.0+3.0 × hdtf+hallo3 (4 GPUs) ─────────────────────
echo ""
echo "=== Batch 2: refseq ==="
run_eval 0 "refseq/cfg1.0/hdtf"   "$ORIG_HDTF"   "demo_out/v2v_eval_cfg_ablation/refseq/cfg1.0/step-3000/hdtf_composited"   &
run_eval 1 "refseq/cfg1.0/hallo3" "$ORIG_HALLO3"  "demo_out/v2v_eval_cfg_ablation/refseq/cfg1.0/step-3000/hallo3_composited" &
run_eval 2 "refseq/cfg3.0/hdtf"   "$ORIG_HDTF"   "demo_out/v2v_eval_cfg_ablation/refseq/cfg3.0/step-3000/hdtf_composited"   &
run_eval 3 "refseq/cfg3.0/hallo3" "$ORIG_HALLO3"  "demo_out/v2v_eval_cfg_ablation/refseq/cfg3.0/step-3000/hallo3_composited" &
wait
echo "  Batch 2 complete."

# ─── Batch 3: LatentSync hdtf+hallo3 (2 GPUs) ───────────────────────────────
echo ""
echo "=== Batch 3: LatentSync ==="
run_eval 0 "latentsync/hdtf"   "$ORIG_HDTF"   "/home/work/.local/qual_results/hdtf_short/LatentSync/final"   &
run_eval 1 "latentsync/hallo3" "$ORIG_HALLO3"  "/home/work/.local/qual_results/hallo3_short/LatentSync/final" &
wait
echo "  Batch 3 complete."

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary (GT-aligned crops) ==="

CSV="${OUTPUT_BASE}/metrics_aligned_crops.csv"
echo "model,cfg,dataset,SSIM,FID,FVD" > "$CSV"

printf "%-12s %-5s %-8s %10s %10s %10s\n" "Model" "CFG" "Dataset" "SSIM" "FID" "FVD"
echo "---------------------------------------------------------------"

for dir in "$OUTPUT_BASE"/*/; do
    [ -d "$dir" ] || continue
    # Recurse into subdirs
    find "$dir" -name "metrics_aligned.log" | while read log; do
        rel=$(echo "$log" | sed "s|${OUTPUT_BASE}/||" | sed 's|/metrics_aligned.log||')

        # Parse model/cfg/dataset from path
        model=$(echo "$rel" | cut -d/ -f1)
        if [ "$model" = "latentsync" ]; then
            cfg="-"
            ds=$(echo "$rel" | cut -d/ -f2)
        else
            cfg=$(echo "$rel" | cut -d/ -f2 | sed 's/cfg//')
            ds=$(echo "$rel" | cut -d/ -f3)
        fi

        ssim=$(grep -oP 'SSIM:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        fid=$(grep -oP 'FID:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")

        printf "%-12s %-5s %-8s %10s %10s %10s\n" "$model" "$cfg" "$ds" "${ssim:--}" "${fid:--}" "${fvd:--}"
        echo "${model},${cfg},${ds},${ssim},${fid},${fvd}" >> "$CSV"
    done
done

echo ""
echo "CSV: $CSV"
echo "Done."
