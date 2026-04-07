"""
Remaster batch encoding script -- VapourSynth + TensorRT.

Decodes any video format, runs DRUNet inference via TensorRT (C++, no GIL),
outputs 10-bit YUV for high-quality HEVC encoding.

Usage:
  vspipe encode.vpy -c y4m -a "input=/path/to/video.mkv" -p - | \
    ffmpeg -i pipe: -c:v hevc_nvenc -preset p4 -cq 18 output.mkv

  See remaster/encode.py for a convenient wrapper that handles the full pipeline.

Arguments (via vspipe -a):
  input     (required)  Path to source video file
  engine    (optional)  Path to pre-built TRT engine file
  hwdecode  (optional)  Hardware decode device: "cuda", "d3d11va", "" for CPU.
                         Default: "" (CPU -- most compatible)

Engine build (one-time, run before first encode):
  tools/vs/vs-plugins/vsmlrt-cuda/trtexec.exe \
    --onnx=checkpoints/drunet_student/drunet_student.onnx \
    --shapes=input:1x3x1080x1920 --fp16 \
    --saveEngine=checkpoints/drunet_student/drunet_student_1080p_fp16.engine
"""
import vapoursynth as vs
import os
import sys

core = vs.core

# ---------------------------------------------------------------------------
# Arguments from vspipe -a "key=value"
# ---------------------------------------------------------------------------
input_path = globals().get("input")
if input_path is None:
    raise ValueError('Missing required argument: -a "input=/path/to/video.mkv"')

hwdecode = globals().get("hwdecode", "")
engine_override = globals().get("engine", None)

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Auto-detect engine: prefer DRUNet student, fall back to NAFNet
if engine_override:
    ENGINE_PATH = engine_override
else:
    candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student",
                     "drunet_student_1080p_fp16.engine"),
    ]
    ENGINE_PATH = None
    for c in candidates:
        if os.path.exists(c):
            ENGINE_PATH = c
            break
    if ENGINE_PATH is None:
        raise FileNotFoundError(
            "No TRT engine found. Build one with:\n"
            '  tools/vs/vs-plugins/vsmlrt-cuda/trtexec.exe '
            '--onnx=checkpoints/drunet_student/drunet_student.onnx '
            '--shapes=input:1x3x1080x1920 --fp16 '
            '--saveEngine=checkpoints/drunet_student/drunet_student_1080p_fp16.engine'
        )

# ---------------------------------------------------------------------------
# Source: decode any format via BestSource, L-SMASH-Works, or ffms2
# ---------------------------------------------------------------------------
source_loaded = False

if not source_loaded:
    try:
        kwargs = {"source": input_path}
        if hwdecode:
            kwargs["hwdevice"] = hwdecode
            kwargs["extrahwframes"] = 9
        clip = core.bs.VideoSource(**kwargs)
        source_loaded = True
    except AttributeError:
        pass

if not source_loaded:
    try:
        clip = core.lsmas.LWLibavSource(source=input_path)
        source_loaded = True
    except AttributeError:
        pass

if not source_loaded:
    try:
        clip = core.ffms2.Source(source=input_path)
        source_loaded = True
    except AttributeError:
        pass

if not source_loaded:
    raise RuntimeError(
        "No source filter available. Install BestSource, L-SMASH-Works, or ffms2."
    )

# ---------------------------------------------------------------------------
# Color space: convert to RGBS (float32 planar RGB) for model input
# ---------------------------------------------------------------------------
orig_w = clip.width
orig_h = clip.height

# Detect color matrix from resolution (709 for HD, 470bg for SD)
matrix = "709" if orig_h > 576 else "470bg"

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

# ---------------------------------------------------------------------------
# Pad to model input size (DRUNet: multiple of 8)
# ---------------------------------------------------------------------------
PAD_FACTOR = 8
TARGET_H = ((orig_h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
TARGET_W = ((orig_w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR

pad_h = TARGET_H - orig_h
pad_w = TARGET_W - orig_w

if pad_h > 0 or pad_w > 0:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

# ---------------------------------------------------------------------------
# Inference: pre-built TRT engine via core.trt.Model
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
# Crop padding and convert back to YUV for encoding
# ---------------------------------------------------------------------------
if pad_h > 0 or pad_w > 0:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

# 10-bit YUV420 output -- preserves quality for HEVC encoding
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)

clip.set_output()
