"""FFmpeg utilities — shared across pipeline and benchmark scripts."""
import os
import subprocess
import re


def get_ffmpeg():
    """Get ffmpeg path — prefer local bin/ (modern, NVENC), fall back to system/imageio."""
    import shutil
    # Prefer project-local ffmpeg (modern build with NVENC)
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError("ffmpeg not found. Install it or pip install imageio-ffmpeg")


def get_ffprobe():
    """Get ffprobe path — prefer local bin/ (modern), fall back to system/imageio."""
    import shutil
    # Prefer project-local ffprobe (matches local ffmpeg build)
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "ffprobe.exe")
    if os.path.exists(local):
        return local
    path = shutil.which("ffprobe")
    if path:
        return path
    # Try alongside whatever ffmpeg we found
    ffmpeg = get_ffmpeg()
    ffdir = os.path.dirname(ffmpeg)
    for name in ("ffprobe", "ffprobe.exe"):
        probe = os.path.join(ffdir, name)
        if os.path.exists(probe):
            return probe
    return None


def get_video_info(path):
    """Get video dimensions, fps, frame count, duration."""
    ffprobe = get_ffprobe()
    if ffprobe:
        try:
            result = _probe_with_ffprobe(ffprobe, path)
            if result[0] > 0:  # got valid width
                return result
        except (IndexError, ValueError):
            pass
    return _probe_with_ffmpeg(get_ffmpeg(), path)


def _probe_with_ffprobe(ffprobe, path):
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
    parts = lines[0].split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_parts = parts[2].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    nb = parts[3] if len(parts) > 3 else ""
    dur = float(lines[-1]) if len(lines) > 1 else 0
    total_frames = int(nb) if nb and nb not in ("N/A", "") else int(dur * fps)
    return w, h, fps, total_frames, dur


def _probe_with_ffmpeg(ffmpeg, path):
    """Fallback: parse 'ffmpeg -i' stderr when ffprobe is unavailable."""
    r = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True)
    info = r.stderr
    # Resolution
    m = re.search(r'(\d{3,5})x(\d{3,5})', info)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)
    # FPS
    fps = 23.976
    m = re.search(r'(\d+(?:\.\d+)?)\s*fps', info)
    if m:
        fps = float(m.group(1))
    # Duration
    duration = 0
    m = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', info)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    total_frames = int(duration * fps)
    return w, h, fps, total_frames, duration


def build_encoder_cmd(ffmpeg, w, h, fps, output_path, encoder, crf):
    """Build the ffmpeg write command for the chosen encoder."""
    fps_str = f"{fps:.6f}"

    # Color space handling for RGB->YUV conversion.
    # The input is raw RGB24 from our model. FFmpeg must convert to YUV for encoding.
    # Without explicit flags, ffmpeg uses BT.601 coefficients for RGB->YUV,
    # but HD content (>576p) requires BT.709. This mismatch causes visible
    # color shifts (reds pushed down, blues pushed up).
    #
    # Fix: use -vf colorspace to force the correct matrix for the conversion,
    # AND set output metadata so players decode with the right matrix.
    is_hd = h > 576
    matrix = "bt709" if is_hd else "smpte170m"

    # Tell the sws scaler which matrix to use for RGB->YUV
    # -sws_flags sets the conversion matrix, colorspace/primaries/trc tag the output
    color_in_flags = [
        "-color_primaries", matrix,
        "-color_trc", matrix,
        "-colorspace", matrix,
    ]
    color_out_flags = [
        "-colorspace", matrix,
        "-color_primaries", matrix,
        "-color_trc", matrix,
    ]

    if encoder == "hevc_nvenc":
        return [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str,
            *color_in_flags,
            "-i", "pipe:0",
            "-c:v", "hevc_nvenc",
            "-preset", "p4", "-tune", "hq",
            "-rc", "vbr", "-cq", str(crf),
            "-pix_fmt", "p010le",  # 10-bit output
            *color_out_flags,
            "-movflags", "+faststart",
            output_path,
        ]
    elif encoder == "libx264":
        return [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str,
            *color_in_flags,
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-crf", str(crf), "-preset", "fast",
            "-pix_fmt", "yuv420p",
            *color_out_flags,
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        return [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str,
            *color_in_flags,
            "-i", "pipe:0",
            "-c:v", "libx265",
            "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p10le",
            *color_out_flags,
            "-x265-params",
            "aq-mode=3:aq-strength=0.8:deblock=-1,-1:no-sao=1:rc-lookahead=40:pools=4",
            "-movflags", "+faststart",
            output_path,
        ]
