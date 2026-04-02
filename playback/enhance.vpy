"""
VapourSynth script for real-time NAFNet denoising during mpv playback.

Loaded by mpv via: vf=vapoursynth=enhance.vpy

Requires:
  - VapourSynth (portable or installed)
  - vs-mlrt plugin with TensorRT backend
  - NAFNet ONNX model exported at 1088x1920

First playback will build a TensorRT engine from ONNX (~2-5 min).
Subsequent playback uses the cached engine.

See playback/README.md for setup instructions.
"""
import vapoursynth as vs
import os

core = vs.core

# Path to the ONNX model — adjust if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                          "nafnet_w32mid4_1088x1920.onnx")

# Input clip from mpv
clip = video_in

# Get dimensions — model expects 1088x1920 (padded 1080p)
orig_w = clip.width
orig_h = clip.height

# Convert to RGBS (32-bit float planar RGB) for model input
# BT.709 matrix for HD content
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# Pad to 1088x1920 if needed (model has fixed input size)
need_pad = (orig_h != 1088 or orig_w != 1920)
if need_pad:
    # Pad bottom and right with reflection
    pad_h = 1088 - orig_h
    pad_w = 1920 - orig_w
    if pad_h > 0 or pad_w > 0:
        clip = core.std.AddBorders(clip, right=max(0, pad_w), bottom=max(0, pad_h))

# Run inference via vs-mlrt TensorRT backend
# fp16=True for RTX 3060 (matches training precision)
# num_streams=1 is fine — model is fast enough for real-time at 1 stream
import vsmlrt
clip = vsmlrt.inference(clip, network_path=MODEL_PATH,
                        backend=vsmlrt.Backend.TRT(
                            fp16=True,
                            num_streams=1,
                            device_id=0,
                        ))

# Crop back to original dimensions if padded
if need_pad:
    clip = core.std.Crop(clip, right=max(0, 1920 - orig_w), bottom=max(0, 1088 - orig_h))

# Convert back to YUV for display
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
