"""
PlainDenoise / UNetDenoise video denoiser — streaming pipeline.

Same streaming architecture as denoise_nafnet.py but for the INT8-native models.
Loads model, fuses reparam blocks, optional torch.compile, streams through ffmpeg.

Usage:
    # UNetDenoise (default)
    python pipelines/denoise_plainnet.py --input data/clip_mid_1080p.mp4 \
        --checkpoint checkpoints/plainnet_unet_nc64_mid2/best.pth --compile

    # PlainDenoise
    python pipelines/denoise_plainnet.py --input data/clip_mid_1080p.mp4 \
        --checkpoint checkpoints/plainnet/best.pth --arch plain --nc 64 --nb 15

    # Quick test (first 100 frames)
    python pipelines/denoise_plainnet.py --input data/clip_mid_1080p.mp4 \
        --checkpoint checkpoints/plainnet_unet_nc64_mid2/best.pth --max-frames 100
"""
import sys
import os
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True
from lib.plainnet_arch import PlainDenoise, UNetDenoise, count_params
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

DEVICE = "cuda"


def load_model(checkpoint_path, arch="unet", nc=64, nb=15,
               nb_enc="2,2", nb_dec="2,2", nb_mid=2,
               device="cuda", fp16=True, use_compile=False):
    """Load PlainDenoise or UNetDenoise from checkpoint."""
    if arch == "unet":
        nb_enc_t = tuple(int(x) for x in nb_enc.split(","))
        nb_dec_t = tuple(int(x) for x in nb_dec.split(","))
        model = UNetDenoise(in_nc=3, nc=nc, nb_enc=nb_enc_t, nb_dec=nb_dec_t,
                           nb_mid=nb_mid, use_bn=True, deploy=False)
    else:
        model = PlainDenoise(in_nc=3, nc=nc, nb=nb, use_bn=True, deploy=False)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=True)

    # Fuse reparam blocks: multi-branch → single conv
    model.fuse_reparam()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    if fp16:
        model = model.half()
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    params = count_params(model)
    print(f"  {arch} nc={nc}: {params/1e3:.1f}K params, "
          f"VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f}GB")

    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    return model


def denoise_video(input_path, output_path, checkpoint_path,
                  arch="unet", nc=64, nb=15, nb_enc="2,2", nb_dec="2,2", nb_mid=2,
                  crf=18, encoder="hevc_nvenc", max_frames=-1,
                  fp16=True, use_compile=False):
    """Stream video through model with threaded IO."""

    model = load_model(checkpoint_path, arch=arch, nc=nc, nb=nb,
                       nb_enc=nb_enc, nb_dec=nb_dec, nb_mid=nb_mid,
                       device=DEVICE, fp16=fp16, use_compile=use_compile)

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)

    # Warmup at video resolution
    if use_compile:
        h_pad = h + (2 - h % 2) % 2  # PlainDenoise only needs even dims
        w_pad = w + (2 - w % 2) % 2
        print(f"  Warming up compiled model at {w_pad}x{h_pad}...")
        dummy = torch.randn(1, 3, h_pad, w_pad, device=DEVICE,
                           dtype=torch.float16 if fp16 else torch.float32)
        dummy = dummy.to(memory_format=torch.channels_last)
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
            _ = model(dummy)
        torch.cuda.synchronize()
        print(f"  Compile warmup: {time.time() - t0:.1f}s")
        del dummy
        torch.cuda.empty_cache()

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    print(f"\nInput: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    # FFmpeg encode command
    ffmpeg_bin = get_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_bytes = w * h * 3
    fps_str = f"{fps:.6f}"

    if encoder == "hevc_nvenc":
        write_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-i", "pipe:0",
            "-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq",
            "-rc", "vbr", "-cq", str(crf), "-pix_fmt", "p010le",
            "-movflags", "+faststart", output_path,
        ]
    else:
        write_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-threads", "0", "-i", "pipe:0",
            "-c:v", "libx265", "-crf", str(crf), "-preset", "fast",
            "-pix_fmt", "yuv420p10le",
            "-x265-params", "aq-mode=3:aq-strength=0.8:deblock=-1,-1:no-sao=1:rc-lookahead=20:pools=4",
            "-movflags", "+faststart", output_path,
        ]

    # PyAV decode + ffmpeg pipe encode
    import av

    frame_queue = Queue(maxsize=16)
    result_queue = Queue(maxsize=16)
    SENTINEL = object()
    WRITER_DONE = object()

    def reader_thread():
        container = av.open(input_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        count = 0
        for frame in container.decode(stream):
            frame_queue.put(frame.to_ndarray(format='rgb24'))
            count += 1
            if max_frames > 0 and count >= max_frames:
                break
        frame_queue.put(SENTINEL)
        container.close()

    writer_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE,
                                   bufsize=frame_bytes * 8)

    def writer_thread():
        while True:
            item = result_queue.get()
            if item is WRITER_DONE:
                break
            writer_proc.stdin.write(item.tobytes())
        writer_proc.stdin.close()
        writer_proc.wait()

    reader_t = Thread(target=reader_thread, daemon=True)
    writer_t = Thread(target=writer_thread, daemon=True)
    reader_t.start()
    writer_t.start()

    print(f"\nProcessing: {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}, encoder={encoder}")

    dtype = torch.float16 if fp16 else torch.float32
    h_pad = h + (2 - h % 2) % 2
    w_pad = w + (2 - w % 2) % 2

    start = time.time()
    processed = 0

    # CUDA events for timing
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    gpu_ms_total = 0.0

    while True:
        frame = frame_queue.get()
        if frame is SENTINEL:
            break

        # Prepare tensor
        t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
        t = t.to(dtype=dtype, device=DEVICE) / 255.0
        if h_pad != h or w_pad != w:
            t = torch.nn.functional.pad(t, (0, w_pad - w, 0, h_pad - h), mode='replicate')
        t = t.to(memory_format=torch.channels_last)

        # Inference
        ev_start.record()
        with torch.no_grad():
            out = model(t)
        ev_end.record()

        # Convert back
        out = out[:, :, :h, :w].clamp(0, 1)
        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().to(torch.uint8)
                  if dtype == torch.uint8
                  else (out.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8))
        result_queue.put(out_np)

        processed += 1
        torch.cuda.synchronize()
        gpu_ms_total += ev_start.elapsed_time(ev_end)

        if processed % 100 == 0 or processed == total_frames:
            elapsed = time.time() - start
            wall_fps = processed / elapsed
            gpu_fps = 1000.0 / (gpu_ms_total / processed)
            print(f"  {processed}/{total_frames} frames | "
                  f"wall: {wall_fps:.1f} fps | "
                  f"gpu: {gpu_fps:.1f} fps ({gpu_ms_total/processed:.1f}ms/frame)")

    result_queue.put(WRITER_DONE)
    reader_t.join()
    writer_t.join()

    elapsed = time.time() - start
    wall_fps = processed / elapsed
    gpu_fps = 1000.0 / (gpu_ms_total / processed) if processed > 0 else 0

    print(f"\nDone: {processed} frames in {elapsed:.1f}s")
    print(f"  Wall: {wall_fps:.1f} fps, GPU: {gpu_fps:.1f} fps ({gpu_ms_total/processed:.1f}ms/frame)")
    print(f"  Output: {output_path} ({os.path.getsize(output_path) / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="PlainDenoise/UNetDenoise video denoiser")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")
    parser.add_argument("--arch", default="unet", choices=["plain", "unet"])
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--nb", type=int, default=15)
    parser.add_argument("--nb-enc", default="2,2")
    parser.add_argument("--nb-dec", default="2,2")
    parser.add_argument("--nb-mid", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--encoder", default="hevc_nvenc",
                        choices=["hevc_nvenc", "libx265"])
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"data/{stem}_unetdenoise.mkv"

    denoise_video(
        args.input, args.output, args.checkpoint,
        arch=args.arch, nc=args.nc, nb=args.nb,
        nb_enc=args.nb_enc, nb_dec=args.nb_dec, nb_mid=args.nb_mid,
        crf=args.crf, encoder=args.encoder, max_frames=args.max_frames,
        fp16=not args.fp32, use_compile=args.compile,
    )


if __name__ == "__main__":
    main()
