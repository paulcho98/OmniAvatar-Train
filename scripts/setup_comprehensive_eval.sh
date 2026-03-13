#!/bin/bash
# Setup unified directory structure for comprehensive evaluation.
# Creates symlinks to GT and all method outputs with consistent naming.
set -euo pipefail

BASE="demo_out/comprehensive_eval"
QUAL="/home/work/.local/qual_results"
GT_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data"
GT_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data"

cd /home/work/.local/OmniAvatar

# ─── GT originals (flat symlinks: {name}_cfr25.mp4 → sub_clip.mp4) ──────────

setup_originals() {
    local ds=$1 gt_dir=$2 out_dir="${BASE}/originals/$1"
    mkdir -p "$out_dir"
    for d in "$gt_dir"/*/; do
        local name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] || continue
        ln -sf "$d/sub_clip.mp4" "$out_dir/${name}.mp4"
    done
    echo "  originals/${ds}: $(ls "$out_dir"/*.mp4 2>/dev/null | wc -l) videos"
}

echo "=== Setting up GT originals ==="
setup_originals hdtf "$GT_HDTF"
setup_originals hallo3 "$GT_HALLO3"

# ─── Method outputs ──────────────────────────────────────────────────────────

link_method() {
    local ds=$1 method=$2 src_dir=$3
    local out_dir="${BASE}/methods/${ds}/${method}"
    mkdir -p "$out_dir"
    for f in "$src_dir"/*.mp4; do
        [ -f "$f" ] || continue
        ln -sf "$(realpath "$f")" "$out_dir/$(basename "$f")"
    done
    local count=$(ls "$out_dir"/*.mp4 2>/dev/null | wc -l)
    echo "  methods/${ds}/${method}: ${count} videos"
}

echo ""
echo "=== Setting up methods ==="

# ─── HDTF baselines ──────────────────────────────────────────────────────────
echo "--- hdtf ---"
link_method hdtf LatentSync "${QUAL}/hdtf_short/LatentSync/final"
link_method hdtf Wav2Lip "${QUAL}/hdtf_short/Wav2Lip"
link_method hdtf VideoReTalking "${QUAL}/hdtf_short/video-retalking"
link_method hdtf Diff2Lip "${QUAL}/hdtf_short/diff2lip"
link_method hdtf MuseTalk "${QUAL}/hdtf_short/MuseTalk/v15"
link_method hdtf Maskall "demo_out/v2v_eval_maskall/step-5500/hdtf_composited"
link_method hdtf RefseqNew "demo_out/v2v_eval_refseq_new/step-3000/hdtf_composited"

# ─── Hallo3 baselines ────────────────────────────────────────────────────────
echo "--- hallo3 ---"
link_method hallo3 LatentSync "${QUAL}/hallo3_short/LatentSync/final"
link_method hallo3 Wav2Lip "${QUAL}/hallo3_short/Wav2Lip"
link_method hallo3 VideoReTalking "${QUAL}/hallo3_short/video-retalking"
link_method hallo3 Diff2Lip "${QUAL}/hallo3_short/diff2lip"
link_method hallo3 MuseTalk "${QUAL}/hallo3_short/MuseTalk"
link_method hallo3 Maskall "demo_out/v2v_eval_maskall/step-5500/hallo3_composited"
link_method hallo3 RefseqNew "demo_out/v2v_eval_refseq_new/step-3000/hallo3_composited"

echo ""
echo "=== Summary ==="
for ds in hdtf hallo3; do
    echo "${ds}:"
    for method_dir in "${BASE}/methods/${ds}"/*/; do
        method=$(basename "$method_dir")
        count=$(ls "$method_dir"/*.mp4 2>/dev/null | wc -l)
        echo "  ${method}: ${count}"
    done
done
