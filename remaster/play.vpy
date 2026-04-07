"""
Remaster real-time playback script -- mpv + VapourSynth + TensorRT.

Enhances video during playback with DRUNet via TensorRT inference.
Uses the same pre-built TRT engine as encode.vpy.

mpv configuration (add to %APPDATA%/mpv/mpv.conf):
  hwdec=auto-copy
  vf=vapoursynth="C:/path/to/remaster/play.vpy"

Toggle enhancement during playback (add to %APPDATA%/mpv/input.conf):
  Ctrl+e vf toggle vapoursynth="C:/path/to/remaster/play.vpy"

Requires pre-built engine (shared with encode.vpy):
  checkpoints/drunet_student/drunet_student_1080p_fp16.engine
"""
import vapoursynth as vs
import os

core = vs.core

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student",
                           "drunet_student_1080p_fp16.engine")
if not os.path.exists(ENGINE_PATH):
    raise FileNotFoundError(
        "No TRT engine found. Build one with:\n"
        '  tools/vs/vs-plugins/vsmlrt-cuda/trtexec.exe '
        '--onnx=checkpoints/drunet_student/drunet_student.onnx '
        '--shapes=input:1x3x1080x1920 --fp16 --useCudaGraph '
        '--saveEngine=checkpoints/drunet_student/drunet_student_1080p_fp16.engine'
    )

# ---------------------------------------------------------------------------
# Input from mpv
# ---------------------------------------------------------------------------
clip = video_in  # noqa: F821 -- provided by mpv's VapourSynth filter

orig_w = clip.width
orig_h = clip.height
matrix = "709" if orig_h > 576 else "470bg"

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

# Pad to multiple of 8 (DRUNet: 3 downsample levels)
PAD_FACTOR = 8
TARGET_H = ((orig_h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
TARGET_W = ((orig_w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
pad_h = TARGET_H - orig_h
pad_w = TARGET_W - orig_w

if pad_h > 0 or pad_w > 0:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

# Real-time: 1 stream for lower latency, CUDA graph for less CPU overhead
clip = core.trt.Model(
    clip,
    engine_path=ENGINE_PATH,
    device_id=0,
    num_streams=1,
    use_cuda_graph=True,
    tilesize=[TARGET_W, TARGET_H],
    overlap=[0, 0],
)

# Crop and convert back for display (8-bit is fine for playback)
if pad_h > 0 or pad_w > 0:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s=matrix)
clip.set_output()
