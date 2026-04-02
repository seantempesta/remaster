"""Fast full-episode denoiser: ffmpeg NVDEC decode → inference → ffmpeg NVENC encode.

Uses ffmpeg for both decode (hevc_cuvid hardware) and encode (h264_nvenc hardware),
with the model running in between via raw frame pipes. This avoids OpenCV's slow
software HEVC decoding which bottlenecks at ~4 fps.

Usage:
    python pipelines/denoise_fast.py --input episode.mkv --output episode_clean.mkv
"""
import argparse
import gc
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

print = partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.nafnet_arch import NAFNet
from lib.paths import PROJECT_ROOT

FFMPEG = str(PROJECT_ROOT / "bin" / "ffmpeg.exe")


def get_video_info(input_path):
    """Get video dimensions, fps, frame count, and codec using ffmpeg."""
    import re
    cmd = [FFMPEG, "-i", input_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    stderr = r.stderr
    # Parse "Duration: HH:MM:SS.ss"
    dur_m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", stderr)
    duration = 0
    if dur_m:
        duration = int(dur_m.group(1))*3600 + int(dur_m.group(2))*60 + float(dur_m.group(3))
    # Parse "Stream #0:0: Video: hevc ... 1920x1080 ... 23.98 fps"
    vid_m = re.search(r"Video:\s+(\w+).*?(\d{3,5})x(\d{3,5}).*?(\d+(?:\.\d+)?)\s+fps", stderr)
    codec = vid_m.group(1) if vid_m else "unknown"
    w = int(vid_m.group(2)) if vid_m else 1920
    h = int(vid_m.group(3)) if vid_m else 1080
    fps = float(vid_m.group(4)) if vid_m else 23.976
    total_frames = int(duration * fps)
    return w, h, fps, total_frames, codec


def run(args):
    device = "cuda"

    # Load model
    enc_blks = [int(x) for x in args.enc_blk_nums.split(",")]
    dec_blks = [int(x) for x in args.dec_blk_nums.split(",")]

    print(f"Loading NAFNet w={args.width} mid={args.middle_blk_num}...")
    model = NAFNet(
        img_channel=3, width=args.width, middle_blk_num=args.middle_blk_num,
        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("params", ckpt.get("model", ckpt))
    del ckpt; gc.collect()
    model.load_state_dict(state); del state; gc.collect()
    model = model.half().to(device).eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    if not args.no_compile:
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    # Get video info
    w, h, fps, total_frames, codec = get_video_info(args.input)
    print(f"Input: {w}x{h}, {fps:.2f} fps, {total_frames} frames ({total_frames/fps/60:.1f} min), codec={codec}")

    # Decoder: ffmpeg with hardware HEVC decode → pipe raw RGB24
    use_hwdec = codec in ("hevc", "h265")
    dec_cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if use_hwdec:
        dec_cmd += ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        print("Using NVDEC hardware decode")
    dec_cmd += [
        "-i", args.input,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "pipe:1",
    ]

    # Encoder: pipe raw RGB24 → ffmpeg h264_nvenc
    video_only = args.output + ".video.mkv"
    enc_cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-rc", "vbr",
        "-cq", str(args.crf),
        "-preset", "p4",
        "-pix_fmt", "yuv420p",
        video_only,
    ]

    # Start both processes
    decoder = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, bufsize=w*h*3*4)
    encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=w*h*3*4)

    frame_size = w * h * 3
    frame_count = 0
    warmup_frames = 5 if not args.no_compile else 0
    t_start = time.time()

    print("Processing...")
    with torch.no_grad():
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            # Raw RGB → tensor
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            inp = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).half().to(device) / 255.0

            out = model(inp).clamp(0, 1)
            out_np = (out.squeeze(0).float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            encoder.stdin.write(out_np.tobytes())
            frame_count += 1

            if frame_count == warmup_frames and warmup_frames > 0:
                t_start = time.time()
                print(f"  Warmup done ({warmup_frames} frames)")

            effective = frame_count - warmup_frames
            if effective > 0 and effective % 500 == 0:
                elapsed = time.time() - t_start
                fps_actual = effective / elapsed
                remaining = (total_frames - frame_count) / fps_actual
                print(f"  {frame_count}/{total_frames} | {fps_actual:.1f} fps | ETA: {remaining/60:.1f} min")

    decoder.stdout.close()
    decoder.wait()
    encoder.stdin.close()
    encoder.wait()

    elapsed = time.time() - t_start
    effective = frame_count - warmup_frames
    fps_actual = effective / elapsed if elapsed > 0 else 0
    print(f"Inference + encode: {frame_count} frames in {elapsed:.0f}s ({fps_actual:.1f} fps)")

    # Mux audio from original
    print("Muxing audio from original...")
    mux_cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_only,
        "-i", args.input,
        "-map", "0:v",
        "-map", "1:a",
        "-c", "copy",
        args.output,
    ]
    subprocess.run(mux_cmd)
    os.remove(video_only)

    size_mb = os.path.getsize(args.output) / 1024**2
    total_time = time.time() - t_start
    print(f"\nDone! {args.output} ({size_mb:.0f} MB)")
    print(f"Total: {total_time/60:.1f} min for {total_frames/fps/60:.1f} min episode ({fps_actual:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Fast episode denoiser")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/nafnet_w32_mid4_dists/nafnet_best.pth")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle-blk-num", type=int, default=4)
    parser.add_argument("--enc-blk-nums", default="2,2,4,8")
    parser.add_argument("--dec-blk-nums", default="2,2,2,2")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
