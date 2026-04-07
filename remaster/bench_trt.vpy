"""
Component-level benchmark for the TRT pipeline.

Tests each stage independently to identify bottlenecks.
Run with: vspipe bench_trt.vpy -p -a "input=... mode=..." .

Modes:
  decode_cpu   - BestSource CPU decode only (no model, no encode)
  decode_gpu   - BestSource GPU decode only (NVDEC)
  colorspace   - CPU decode + YUV->RGB + RGB->YUV (no model)
  inference    - CPU decode + color convert + TRT model (output RGBS)
  full         - Complete pipeline (decode + color + model + output YUV)
"""
import vapoursynth as vs
import os

core = vs.core

input_path = globals().get("input")
if input_path is None:
    raise ValueError('Missing: -a "input=/path/to/video.mkv"')

mode = globals().get("mode", "full")

SCRIPT_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Source decode
# ---------------------------------------------------------------------------
use_gpu_decode = mode in ("decode_gpu",)

if use_gpu_decode:
    clip = core.bs.VideoSource(source=input_path, hwdevice="cuda", extrahwframes=9)
else:
    clip = core.bs.VideoSource(source=input_path)

orig_w = clip.width
orig_h = clip.height
matrix = "709" if orig_h > 576 else "470bg"

if mode in ("decode_cpu", "decode_gpu"):
    # Just decode + minimal format conversion for output
    clip = core.resize.Bicubic(clip, format=vs.YUV420P10,
                               matrix_in_s=matrix, matrix_s=matrix)
    clip.set_output()

elif mode == "colorspace":
    # Decode + YUV->RGBS + pad + unpad + RGBS->YUV (no model)
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

    PAD_FACTOR = 8
    TARGET_H = ((orig_h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
    TARGET_W = ((orig_w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
    pad_h = TARGET_H - orig_h
    pad_w = TARGET_W - orig_w
    if pad_h > 0 or pad_w > 0:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)
        clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

    clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)
    clip.set_output()

elif mode == "inference":
    # Decode + color convert + TRT model (output RGBS, skip final YUV)
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

    PAD_FACTOR = 8
    TARGET_H = ((orig_h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
    TARGET_W = ((orig_w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
    pad_h = TARGET_H - orig_h
    pad_w = TARGET_W - orig_w
    if pad_h > 0 or pad_w > 0:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

    ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student",
                               "drunet_student_1080p_fp16.engine")
    clip = core.trt.Model(
        clip, engine_path=ENGINE_PATH, device_id=0,
        num_streams=4, use_cuda_graph=True,
        tilesize=[TARGET_W, TARGET_H], overlap=[0, 0],
    )
    clip.set_output()

else:
    # Full pipeline: decode + color + model + YUV output
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

    PAD_FACTOR = 8
    TARGET_H = ((orig_h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
    TARGET_W = ((orig_w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
    pad_h = TARGET_H - orig_h
    pad_w = TARGET_W - orig_w
    if pad_h > 0 or pad_w > 0:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

    ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student",
                               "drunet_student_1080p_fp16.engine")
    clip = core.trt.Model(
        clip, engine_path=ENGINE_PATH, device_id=0,
        num_streams=4, use_cuda_graph=True,
        tilesize=[TARGET_W, TARGET_H], overlap=[0, 0],
    )

    if pad_h > 0 or pad_w > 0:
        clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

    clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)
    clip.set_output()
