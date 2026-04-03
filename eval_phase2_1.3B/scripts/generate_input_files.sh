#!/bin/bash
# Generate input txt files for V2V eval from validation_data/ directory.
# Each line: prompt@@video_path@@audio_path
#
# Usage: bash scripts/generate_input_files.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

INPUT_DIR="inputs"
mkdir -p "$INPUT_DIR"

generate_input_file() {
    local ds=$1
    local data_dir="validation_data/${ds}"
    local out_file="${INPUT_DIR}/${ds}_latentsync.txt"

    if [ ! -d "$data_dir" ]; then
        echo "ERROR: ${data_dir} not found"
        return 1
    fi

    > "$out_file"
    local count=0
    for d in "$data_dir"/*/; do
        [ -d "$d" ] || continue
        local name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue

        local video="$d/sub_clip.mp4"
        local audio="$d/audio.wav"
        local prompt_file="$d/prompt.txt"

        [ -f "$video" ] || continue
        [ -f "$audio" ] || continue

        local prompt="a person is talking"
        if [ -f "$prompt_file" ]; then
            prompt=$(cat "$prompt_file")
        fi

        # Use absolute paths (required by inference_v2v.py)
        echo "${prompt}@@$(realpath "$video")@@$(realpath "$audio")" >> "$out_file"
        count=$((count + 1))
    done
    echo "  ${ds}: ${count} samples -> ${out_file}"
}

echo "=== Generating input files ==="
generate_input_file hdtf
generate_input_file hallo3
echo "Done."
