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

Engines are picked by source resolution; build one for a new resolution with
  python tools/build_trt_engine.py --width 1920 --height 1080
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
# Arguments from vspipe -a "key=value"
# ---------------------------------------------------------------------------
input_path = globals().get("input")
if input_path is None:
    raise ValueError('Missing required argument: -a "input=/path/to/video.mkv"')

hwdecode = globals().get("hwdecode", "")
engine_override = globals().get("engine", None)
if engine_override:
    os.environ["REMASTER_ENGINE"] = engine_override

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
# Inference + output
# ---------------------------------------------------------------------------
matrix = vsr.matrix_for(clip)
clip = vsr.enhance(clip, num_streams=4)

# 10-bit YUV420 output -- preserves quality for HEVC encoding
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)

clip.set_output()
