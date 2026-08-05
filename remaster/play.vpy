"""
Remaster real-time playback script -- mpv + VapourSynth + TensorRT.

Enhances video during playback with DRUNet via TensorRT inference.
Uses the same pre-built TRT engines as encode.vpy, picked by source resolution.

mpv configuration (add to %APPDATA%/mpv/mpv.conf):
  hwdec=auto-copy
  vf=vapoursynth="C:/path/to/remaster/play.vpy"

Toggle enhancement during playback (add to %APPDATA%/mpv/input.conf):
  Ctrl+e vf toggle vapoursynth="C:/path/to/remaster/play.vpy"

If mpv's VapourSynth filter does not expose __file__, set REMASTER_DIR to this
directory so the script can find vs_remaster.py.
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

# ---------------------------------------------------------------------------
# Input from mpv
# ---------------------------------------------------------------------------
clip = video_in  # noqa: F821 -- provided by mpv's VapourSynth filter

matrix = vsr.matrix_for(clip)

# Real-time: 1 stream for lower latency, CUDA graph for less CPU overhead
clip = vsr.enhance(clip, num_streams=1)

# 8-bit is fine for display
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s=matrix)
clip.set_output()
