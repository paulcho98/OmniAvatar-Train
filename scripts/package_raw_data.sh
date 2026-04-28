#!/bin/bash
# =============================================================================
# Package raw training data for transfer to a new machine.
#
# Creates a self-contained bundle (~22 GB) containing:
#   1. Raw source videos (H.264 clips, ~430 KB each)
#   2. Audio .wav files (16 kHz mono PCM, ~230 KB each)
#   3. Metadata CSV + data lists + LatentSync mask
#   4. Prompt texts (for T5 re-encoding on the target)
#
# Text embeddings are NOT included — they are re-encoded from prompts on
# the target machine using scripts/precompute_text_embeddings.py.
#
# Usage:
#   bash scripts/package_raw_data.sh /path/to/output_bundle/
#
# On the target machine, follow docs/preprocessing.md to unpack and
# precompute all derived data.
# =============================================================================

set -euo pipefail

BUNDLE_DIR="${1:?Usage: $0 <output_bundle_dir>}"
mkdir -p "$BUNDLE_DIR"

DATA_ROOT="/home/work/stableavatar_data/v2v_training_data"
VIDEO_SOURCE="/home/work/.local/combined_data/high_visual_quality"
MASK_PATH="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
COMBINED_LIST="$DATA_ROOT/video_square_path_combined.txt"

echo "============================================="
echo "  Packaging raw training data"
echo "============================================="
echo "  Bundle dir: $BUNDLE_DIR"
echo "  Data root:  $DATA_ROOT"
echo "  Video src:  $VIDEO_SOURCE"
echo "============================================="
echo ""

# -----------------------------------------------------------------
# 1. Videos: collect only the ones referenced by the combined list
# -----------------------------------------------------------------
echo "[1/5] Collecting referenced videos..."
VIDEO_LIST=$(mktemp)
cat "$COMBINED_LIST" | while read d; do
    target=$(readlink -f "$d/sub_clip.mp4" 2>/dev/null)
    [ -f "$target" ] && echo "$target"
done | sort -u > "$VIDEO_LIST"

NUM_VIDEOS=$(wc -l < "$VIDEO_LIST")
echo "  $NUM_VIDEOS unique videos found"

echo "  Creating videos.tar (this may take a few minutes)..."
tar cf "$BUNDLE_DIR/videos.tar" -T "$VIDEO_LIST" \
    --transform='s|.*/high_visual_quality/|high_visual_quality/|'
echo "  Done: $(du -sh "$BUNDLE_DIR/videos.tar" | cut -f1)"
rm "$VIDEO_LIST"

# -----------------------------------------------------------------
# 2. Audio: collect audio.wav from each sample directory
# -----------------------------------------------------------------
echo ""
echo "[2/5] Collecting audio files..."
AUDIO_LIST=$(mktemp)
cat "$COMBINED_LIST" | while read d; do
    wav="$d/audio.wav"
    [ -f "$wav" ] && echo "$wav"
done > "$AUDIO_LIST"

NUM_AUDIO=$(wc -l < "$AUDIO_LIST")
echo "  $NUM_AUDIO audio files found"

echo "  Creating audio.tar.gz..."
# Strip the data root prefix so paths are relative
tar czf "$BUNDLE_DIR/audio.tar.gz" -T "$AUDIO_LIST" \
    --transform="s|$DATA_ROOT/||"
echo "  Done: $(du -sh "$BUNDLE_DIR/audio.tar.gz" | cut -f1)"
rm "$AUDIO_LIST"

# -----------------------------------------------------------------
# 3. Prompts: collect prompt.txt from each sample directory
# -----------------------------------------------------------------
echo ""
echo "[3/5] Collecting prompt files..."
PROMPT_LIST=$(mktemp)
cat "$COMBINED_LIST" | while read d; do
    p="$d/prompt.txt"
    [ -f "$p" ] && echo "$p"
done > "$PROMPT_LIST"

NUM_PROMPTS=$(wc -l < "$PROMPT_LIST")
echo "  $NUM_PROMPTS prompt files found"

echo "  Creating prompts.tar.gz..."
tar czf "$BUNDLE_DIR/prompts.tar.gz" -T "$PROMPT_LIST" \
    --transform="s|$DATA_ROOT/||"
echo "  Done: $(du -sh "$BUNDLE_DIR/prompts.tar.gz" | cut -f1)"
rm "$PROMPT_LIST"

# -----------------------------------------------------------------
# 4. Metadata: CSV, data lists, mask
# -----------------------------------------------------------------
echo ""
echo "[4/5] Copying metadata files..."
mkdir -p "$BUNDLE_DIR/metadata"

cp "$DATA_ROOT/metadata_combined_with_voxceleb2.csv" "$BUNDLE_DIR/metadata/"
cp "$DATA_ROOT/video_square_path.txt" "$BUNDLE_DIR/metadata/"
cp "$DATA_ROOT/video_square_path_combined.txt" "$BUNDLE_DIR/metadata/"
cp "$DATA_ROOT/video_square_val10.txt" "$BUNDLE_DIR/metadata/"
[ -f "$DATA_ROOT/video_square_val2_test.txt" ] && \
    cp "$DATA_ROOT/video_square_val2_test.txt" "$BUNDLE_DIR/metadata/"
cp "$MASK_PATH" "$BUNDLE_DIR/metadata/mask.png"

echo "  Done"

# -----------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------
echo ""
echo "[5/5] Bundle complete!"
echo ""
echo "============================================="
echo "  Contents of $BUNDLE_DIR:"
du -sh "$BUNDLE_DIR"/*
echo ""
echo "  Total: $(du -sh "$BUNDLE_DIR" | cut -f1)"
echo "============================================="
echo ""
echo "Next steps on the target machine:"
echo "  1. Unpack (see docs/preprocessing.md Step 1)"
echo "  2. Precompute text embeddings (Step 3)"
echo "  3. Precompute VAE/audio/ref latents with --legacy_twostep_mask (Step 4)"
echo "  4. Update data list paths (Step 5)"
echo "  5. Launch training (Step 6)"
