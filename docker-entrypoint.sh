#!/bin/bash
# docker-entrypoint.sh -- Auto-setup and run the remaster pipeline
#
# Supports single files and batch processing. TRT engines are built once
# per resolution and cached in /app/engines (mount a volume for persistence).
#
# Usage:
#   remaster input.mkv output.mkv [--cq 20] [--preset p5]
#   remaster --batch /data/input/ /data/output/ [--cq 20]
set -euo pipefail

ONNX_PATH="/app/model/drunet_student.onnx"
ENGINE_DIR="/app/engines"
BINARY="/app/bin/remaster_pipeline"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[remaster] $*" >&2; }
die() { echo "[remaster] ERROR: $*" >&2; exit 1; }

detect_resolution() {
    local input="$1"
    # Use key=value format (robust against DV/HDR extra fields that break CSV)
    local width height
    width=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width -of default=noprint_wrappers=1:nokey=1 "$input" 2>/dev/null) || true
    height=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=height -of default=noprint_wrappers=1:nokey=1 "$input" 2>/dev/null) || true
    if [[ -n "$width" && -n "$height" ]]; then
        echo "${width}x${height}"
    fi
}

pad_to_8() {
    echo $(( ($1 + 7) / 8 * 8 ))
}

get_or_build_engine() {
    local width=$1 height=$2
    local pad_w=$(pad_to_8 "$width")
    local pad_h=$(pad_to_8 "$height")
    local engine="${ENGINE_DIR}/drunet_${pad_h}p_fp16.engine"

    if [[ -f "$engine" ]]; then
        log "Engine cached: ${pad_w}x${pad_h}"
    else
        log "Building TRT engine for ${pad_w}x${pad_h} (one-time, ~2 min)..."
        trtexec \
            --onnx="$ONNX_PATH" \
            --shapes=input:1x3x${pad_h}x${pad_w} \
            --fp16 \
            --inputIOFormats=fp16:chw \
            --outputIOFormats=fp16:chw \
            --useCudaGraph \
            --saveEngine="$engine" \
            > /dev/null 2>&1

        [[ -f "$engine" ]] || die "Engine build failed. Run with --entrypoint bash to debug."
        log "Engine built and cached: $engine"
    fi
    echo "$engine"
}

process_file() {
    local input="$1" output="$2"
    shift 2
    local extra_args=("$@")

    [[ -f "$input" ]] || { log "SKIP: not found: $input"; return 1; }

    local dims
    dims=$(detect_resolution "$input")
    [[ -n "$dims" && "$dims" != *"N/A"* ]] || { log "SKIP: can't detect resolution: $input"; return 1; }

    local width=${dims%%x*}
    local height=${dims##*x}

    # Get or build engine (cached by resolution)
    local engine
    engine=$(get_or_build_engine "$width" "$height")

    log "$(basename "$input") (${width}x${height}) -> $(basename "$output")"
    "$BINARY" --input "$input" --output "$output" --engine "$engine" "${extra_args[@]}"
}

print_usage() {
    cat >&2 <<'USAGE'
Remaster -- GPU video enhancement pipeline

Usage:
  remaster INPUT OUTPUT [OPTIONS]         Process a single file
  remaster --batch INPUT_DIR OUTPUT_DIR [OPTIONS]  Process all videos in a directory

Options (passed to the C++ pipeline):
  --cq N          Constant quality (0-51, lower=better, default: 24)
  --preset pN     NVENC preset p1-p7 (default: p4)
  --no-audio      Skip audio/subtitle passthrough
  --color-transfer  Match output color/brightness to input (for color shifts)

Examples:
  remaster /data/episode.mkv /data/enhanced/episode.mkv
  remaster /data/episode.mkv /data/enhanced/episode.mkv --cq 20
  remaster --batch /data/originals/ /data/enhanced/

On first run, a TensorRT engine is built for your GPU (~2 min). This is
cached automatically -- subsequent runs start processing immediately.
USAGE
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

# Batch mode: process all video files in a directory
if [[ "$1" == "--batch" ]]; then
    [[ $# -ge 3 ]] || { print_usage; exit 1; }
    INPUT_DIR="$2"
    OUTPUT_DIR="$3"
    shift 3
    EXTRA_ARGS=("$@")

    [[ -d "$INPUT_DIR" ]] || die "Input directory not found: $INPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Find video files (common extensions)
    mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f \
        \( -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.avi" -o -iname "*.ts" -o -iname "*.m2ts" \) \
        | sort)

    [[ ${#FILES[@]} -gt 0 ]] || die "No video files found in $INPUT_DIR"

    log "Batch: ${#FILES[@]} files in $(basename "$INPUT_DIR")"

    PROCESSED=0
    FAILED=0
    START_TIME=$SECONDS

    for input in "${FILES[@]}"; do
        base=$(basename "$input")
        # Output as MKV regardless of input format
        output="${OUTPUT_DIR}/${base%.*}.mkv"

        if process_file "$input" "$output" "${EXTRA_ARGS[@]}"; then
            ((PROCESSED++))
        else
            ((FAILED++))
        fi
        echo "" >&2
    done

    ELAPSED=$(( SECONDS - START_TIME ))
    log "Batch complete: ${PROCESSED} processed, ${FAILED} failed, ${ELAPSED}s total"
    exit 0
fi

# Single file mode
INPUT="$1"
OUTPUT="$2"
shift 2
EXTRA_ARGS=("$@")

[[ -f "$INPUT" ]] || die "Input file not found: $INPUT"

# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT")
[[ -d "$OUTPUT_DIR" ]] || mkdir -p "$OUTPUT_DIR"

process_file "$INPUT" "$OUTPUT" "${EXTRA_ARGS[@]}"
