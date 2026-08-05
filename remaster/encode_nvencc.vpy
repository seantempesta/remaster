"""
Remaster encoding script for NVEncC -- VapourSynth + TensorRT.

NVEncC runs VapourSynth in-process (no pipe), eliminating the y4m bottleneck.

Usage:
  set REMASTER_INPUT=C:/path/to/video.mkv
  NVEncC64.exe --vpy --vpy-mt --vsdir tools/vs -i remaster/encode_nvencc.vpy \
    --codec hevc --profile main10 --output-depth 10 --preset p4 --vbr-quality 18 \
    --audio-copy -o output.mkv

Or use remaster/encode_nvencc.py for a convenient wrapper.

Environment:
  REMASTER_INPUT         (required) source video path
  REMASTER_ENGINE        (optional) TRT engine to use, overriding auto-selection
  REMASTER_DIR           (optional) path to this directory, if __file__ is unavailable
  REMASTER_TRIM_START    (optional) first frame to encode
  REMASTER_TRIM_LENGTH   (optional) how many frames to encode from there
"""
import os
import sys

import vapoursynth as vs

core = vs.core


def _remaster_dir():
    d = os.environ.get("REMASTER_DIR")
    if d:
        return d
    d = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    return d if os.path.exists(os.path.join(d, "vs_remaster.py")) else os.path.join(d, "remaster")


sys.path.insert(0, _remaster_dir())
import vs_remaster as vsr  # noqa: E402

input_path = os.environ.get("REMASTER_INPUT")
if input_path is None:
    raise ValueError("Set REMASTER_INPUT environment variable to the video path")

# ---------------------------------------------------------------------------
# Source decode
# ---------------------------------------------------------------------------
source_loaded = False

# Prefer L-SMASH-Works for long files (no full-file indexing needed)
if not source_loaded:
    try:
        clip = core.lsmas.LWLibavSource(source=input_path)
        source_loaded = True
    except AttributeError:
        pass

# Fall back to BestSource (accurate but slow to index long files)
if not source_loaded:
    try:
        clip = core.bs.VideoSource(source=input_path)
        source_loaded = True
    except AttributeError:
        pass

if not source_loaded:
    raise RuntimeError("No source filter available (L-SMASH-Works or BestSource)")

# ---------------------------------------------------------------------------
# Optional trim, for spot checks. This has to happen in the script: NVEncC's
# --seek is silently ignored with --vpy input, so it would encode from frame 0.
# ---------------------------------------------------------------------------
trim_start = os.environ.get("REMASTER_TRIM_START")
if trim_start:
    length = int(os.environ.get("REMASTER_TRIM_LENGTH", "300"))
    start = min(int(trim_start), max(0, clip.num_frames - 1))
    clip = core.std.Trim(clip, first=start, length=min(length, clip.num_frames - start))

# ---------------------------------------------------------------------------
# Inference + output (NVEncC handles the YUV -> HEVC encode)
# ---------------------------------------------------------------------------
matrix = vsr.matrix_for(clip)
clip = vsr.enhance(clip, num_streams=4)

# 10-bit YUV420 output -- preserves quality for HEVC encoding
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)

clip.set_output()
