#!/bin/bash
# Re-composite all evaluation videos with fixed pipeline.
# Archives old composited dirs as {ds}_composited_old/ and creates new ones.
#
# Usage: bash scripts/recomposite_all_fixed.sh

set -e
cd /home/work/.local/OmniAvatar

PYTHON="/home/work/.local/miniconda3/envs/omniavatar/bin/python"
SCRIPT="scripts/recomposite_fixed.py"
MASK="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"

# Face cache directories (shared across all experiments)
FC_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache"
FC_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache"
FC_MIXED="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/face_cache"

# ─── Helper function ─────────────────────────────────────────────────────────

recomposite() {
    local gpu=$1
    local exp_base=$2      # e.g., demo_out/v2v_eval_maskall
    local step=$3          # e.g., step-5500
    local ds=$4            # e.g., hdtf
    local fc_dir=$5        # face cache dir
    local input_suffix=$6  # e.g., "_composited" or ""

    local step_dir="${exp_base}/${step}"
    local comp_dir="${step_dir}/${ds}_composited"
    local old_dir="${step_dir}/${ds}_composited_old"

    # Determine input directory
    if [ -d "${step_dir}/${ds}_aligned" ] && [ -n "$(ls ${step_dir}/${ds}_aligned/*.mp4 2>/dev/null)" ]; then
        local input_dir="${step_dir}/${ds}_aligned"
        input_suffix=""
        echo "  [GPU $gpu] ${step}/${ds}: using aligned files from ${ds}_aligned/"
    else
        local input_dir="${step_dir}/${ds}"
        echo "  [GPU $gpu] ${step}/${ds}: using raw files from ${ds}/ (suffix: ${input_suffix})"
    fi

    # Check input exists
    if [ ! -d "$input_dir" ] || [ -z "$(ls ${input_dir}/*.mp4 2>/dev/null)" ]; then
        echo "  [GPU $gpu] [SKIP] ${step}/${ds} (no input files)"
        return
    fi

    # Archive old composited if not already done
    if [ -d "$comp_dir" ] && [ ! -d "$old_dir" ]; then
        echo "  [GPU $gpu] Archiving ${comp_dir} → ${old_dir}"
        mv "$comp_dir" "$old_dir"
    elif [ -d "$comp_dir" ] && [ -d "$old_dir" ]; then
        echo "  [GPU $gpu] Old archive exists, removing current composited"
        rm -rf "$comp_dir"
    fi

    # Re-composite
    local suffix_arg=""
    [ -n "$input_suffix" ] && suffix_arg="--input_suffix ${input_suffix}"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT \
        --aligned_dir "$input_dir" \
        --originals_dir "${exp_base}/originals/${ds}" \
        --face_cache_dir "$fc_dir" \
        --output_dir "$comp_dir" \
        --mask_path "$MASK" \
        --device cuda:0 \
        $suffix_arg

    # Verify count
    local expected=$(ls "${exp_base}/originals/${ds}"/*.mp4 2>/dev/null | wc -l)
    local actual=$(ls "${comp_dir}"/*.mp4 2>/dev/null | wc -l)
    echo "  [GPU $gpu] ${step}/${ds}: ${actual}/${expected} videos composited"
}

# ─── Refseq_new (has aligned files) ──────────────────────────────────────────

echo "=========================================="
echo "=== REFSEQ_NEW (step-3000, 3 datasets) ==="
echo "=========================================="

# Run 3 datasets on 3 GPUs in parallel
recomposite 0 demo_out/v2v_eval_refseq_new step-3000 hdtf "$FC_HDTF" "" &
recomposite 1 demo_out/v2v_eval_refseq_new step-3000 hallo3 "$FC_HALLO3" "" &
recomposite 2 demo_out/v2v_eval_refseq_new step-3000 hallo3_mixed "$FC_MIXED" "" &
wait
echo ""

# ─── Refseq (no aligned files, use raw _composited.mp4) ─────────────────────

echo "=========================================="
echo "=== REFSEQ (step-1500, 3 datasets) ==="
echo "=========================================="

recomposite 0 demo_out/v2v_eval_refseq step-1500 hdtf "$FC_HDTF" "_composited" &
recomposite 1 demo_out/v2v_eval_refseq step-1500 hallo3 "$FC_HALLO3" "_composited" &
recomposite 2 demo_out/v2v_eval_refseq step-1500 hallo3_mixed "$FC_MIXED" "_composited" &
wait
echo ""

# ─── Maskall (no aligned files, use raw _composited.mp4) ────────────────────

for step in step-3000 step-4000 step-4500 step-5500; do
    echo "=========================================="
    echo "=== MASKALL (${step}, 3 datasets) ==="
    echo "=========================================="

    recomposite 0 demo_out/v2v_eval_maskall "$step" hdtf "$FC_HDTF" "_composited" &
    recomposite 1 demo_out/v2v_eval_maskall "$step" hallo3 "$FC_HALLO3" "_composited" &
    recomposite 2 demo_out/v2v_eval_maskall "$step" hallo3_mixed "$FC_MIXED" "_composited" &
    wait
    echo ""
done

echo "=========================================="
echo "=== ALL DONE ==="
echo "=========================================="

# Verify all outputs match original resolutions
echo ""
echo "Spot-checking output resolutions..."
for exp in v2v_eval_refseq_new v2v_eval_refseq v2v_eval_maskall; do
    for step_dir in demo_out/${exp}/step-*/; do
        [ -d "$step_dir" ] || continue
        step=$(basename "$step_dir")
        for ds in hdtf hallo3 hallo3_mixed; do
            comp="${step_dir}/${ds}_composited"
            orig="demo_out/${exp}/originals/${ds}"
            [ -d "$comp" ] || continue
            # Check first video
            first_comp=$(ls "$comp"/*.mp4 2>/dev/null | head -1)
            [ -z "$first_comp" ] && continue
            stem=$(basename "$first_comp" .mp4)
            first_orig="${orig}/${stem}.mp4"
            [ -f "$first_orig" ] || continue
            comp_res=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height "$first_comp" | grep -E "width|height" | tr '\n' ' ')
            orig_res=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height "$first_orig" | grep -E "width|height" | tr '\n' ' ')
            if [ "$comp_res" = "$orig_res" ]; then
                echo "  ✓ ${exp}/${step}/${ds}: resolution matches"
            else
                echo "  ✗ ${exp}/${step}/${ds}: MISMATCH comp=${comp_res} orig=${orig_res}"
            fi
        done
    done
done
