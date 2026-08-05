"""Shared VapourSynth inference helpers for the remaster .vpy scripts.

TRT engines are built at a static shape and a fixed IO precision, so the clip
handed to core.trt.Model has to match both exactly. Getting either wrong fails
at runtime ("bits per sample mismatch" / a shape assertion), so the logic lives
here once instead of being copy-pasted into every .vpy.

Engine naming convention: drunet_student_{W}x{H}_{precision}.engine
  e.g. drunet_student_1920x816_fp16.engine

Build one for a new resolution with tools/build_trt_engine.py.
"""
import os
import re

import vapoursynth as vs

core = vs.core

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student")

# Built before the {W}x{H} convention existed; still the padding fallback.
LEGACY_1080P = "drunet_student_1080p_fp16.engine"


def engine_shape(engine_path):
    """(width, height) the engine was built for, parsed from its filename."""
    name = os.path.basename(engine_path)
    m = re.search(r"_(\d+)x(\d+)_", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    if "_1080p_" in name:
        return 1920, 1080
    raise ValueError(
        f"Cannot determine input shape of {name}. Engines must be named "
        f"drunet_student_{{W}}x{{H}}_{{precision}}.engine"
    )


def engine_format(engine_path):
    """VapourSynth RGB format matching the engine's IO precision.

    An engine built with --inputIOFormats=fp16:chw needs RGBH (16-bit half);
    fp32 IO (including INT8 engines, whose IO stays fp32) needs RGBS.
    """
    return vs.RGBH if "_fp16" in os.path.basename(engine_path) else vs.RGBS


def resolve_engine(width, height):
    """Pick the engine to use for a source of this size.

    Prefers one built for the source's own resolution -- letterboxed content is
    a lot of pixels short of 1080p, so a native engine both avoids padding and
    runs faster (1920x816 is ~55 fps vs ~43 fps padded up to 1920x1080).

    REMASTER_ENGINE overrides the search entirely.
    """
    override = os.environ.get("REMASTER_ENGINE")
    if override:
        return override

    native = os.path.join(ENGINE_DIR, f"drunet_student_{width}x{height}_fp16.engine")
    if os.path.exists(native):
        return native
    return os.path.join(ENGINE_DIR, LEGACY_1080P)


def matrix_for(clip):
    """Color matrix to use for a clip (709 for HD, 470bg for SD)."""
    return "709" if clip.height > 576 else "470bg"


def enhance(clip, num_streams=4):
    """Run the remaster model over a YUV clip. Returns RGB at the source size.

    Callers convert the result back to whatever output format they need.
    """
    orig_w, orig_h = clip.width, clip.height

    engine_path = resolve_engine(orig_w, orig_h)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"TRT engine not found: {engine_path}\n"
            f"Build one with: python tools/build_trt_engine.py "
            f"--width {orig_w} --height {orig_h}"
        )

    target_w, target_h = engine_shape(engine_path)
    if orig_w > target_w or orig_h > target_h:
        raise ValueError(
            f"Source {orig_w}x{orig_h} is larger than engine {target_w}x{target_h}. "
            f"Build a matching engine: python tools/build_trt_engine.py "
            f"--width {orig_w} --height {orig_h}"
        )

    clip = core.resize.Bicubic(clip, format=engine_format(engine_path),
                               matrix_in_s=matrix_for(clip))

    pad_w = target_w - orig_w
    pad_h = target_h - orig_h
    if pad_w or pad_h:
        # Edge-replicate, not black borders -- zero fill puts a hard edge inside
        # the model's receptive field and leaves a dark seam along the boundary.
        # Requesting a source window larger than the frame makes zimg clamp to
        # the edge pixel; width == src_width keeps it 1:1 (no resampling).
        clip = core.resize.Point(clip, width=target_w, height=target_h,
                                 src_width=target_w, src_height=target_h)

    clip = core.trt.Model(
        clip,
        engine_path=engine_path,
        device_id=0,
        num_streams=num_streams,
        use_cuda_graph=True,
        tilesize=[target_w, target_h],
        overlap=[0, 0],
    )

    if pad_w or pad_h:
        clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

    return clip
