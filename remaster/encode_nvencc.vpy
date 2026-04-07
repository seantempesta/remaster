"""
Remaster encoding script for NVEncC -- VapourSynth + TensorRT.

NVEncC runs VapourSynth in-process (no pipe), eliminating the y4m bottleneck.

Usage:
  set REMASTER_INPUT=C:/path/to/video.mkv
  NVEncC64.exe --vpy --vpy-mt --vsdir tools/vs -i remaster/encode_nvencc.vpy \
    --codec hevc --profile main10 --output-depth 10 --preset p4 --vbr-quality 18 \
    --audio-copy -o output.mkv

Or use remaster/encode_nvencc.py for a convenient wrapper.
"""
import vapoursynth as vs
import os

core = vs.core

# NVEncC passes input via environment variable (set by wrapper script)
input_path = os.environ.get("REMASTER_INPUT")
if input_path is None:
    raise ValueError("Set REMASTER_INPUT environment variable to the video path")

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student",
                           "drunet_student_1080p_fp16.engine")
if not os.path.exists(ENGINE_PATH):
    raise FileNotFoundError(f"TRT engine not found: {ENGINE_PATH}")

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
# Color space + padding
# ---------------------------------------------------------------------------
orig_w = clip.width
orig_h = clip.height
matrix = "709" if orig_h > 576 else "470bg"

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

PAD_FACTOR = 8
TARGET_H = ((orig_h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
TARGET_W = ((orig_w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
pad_h = TARGET_H - orig_h
pad_w = TARGET_W - orig_w

if pad_h > 0 or pad_w > 0:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

# ---------------------------------------------------------------------------
# TRT inference
# ---------------------------------------------------------------------------
clip = core.trt.Model(
    clip,
    engine_path=ENGINE_PATH,
    device_id=0,
    num_streams=4,
    use_cuda_graph=True,
    tilesize=[TARGET_W, TARGET_H],
    overlap=[0, 0],
)

# ---------------------------------------------------------------------------
# Crop + output (NVEncC handles YUV conversion internally)
# ---------------------------------------------------------------------------
if pad_h > 0 or pad_w > 0:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

# Output as YUV420P10 for HEVC encoding
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)

clip.set_output()
