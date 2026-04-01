"""
NAFNet video denoiser — streaming pipeline for distilled NAFNet model.

Same streaming architecture as denoise_batch.py but using NAFNet instead of SCUNet.
NAFNet is a pure CNN (no attention), so it's faster and torch.compile friendly.

Usage:
    # With distilled checkpoint
    python pipelines/denoise_nafnet.py --input episode.mkv --checkpoint checkpoints/nafnet_distill/nafnet_best.pth

    # With SIDD pretrained (for comparison)
    python pipelines/denoise_nafnet.py --input episode.mkv --checkpoint reference-code/NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth

    # With torch.compile for maximum speed
    python pipelines/denoise_nafnet.py --input episode.mkv --checkpoint checkpoints/nafnet_distill/nafnet_best.pth --compile

    # Quick test
    python pipelines/denoise_nafnet.py --input episode.mkv --max-frames 100 --checkpoint checkpoints/nafnet_distill/nafnet_best.pth
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
from lib.nafnet_arch import NAFNet
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

DEVICE = "cuda"


def load_nafnet(checkpoint_path, device="cuda", fp16=True, use_compile=False):
    """Load NAFNet-width64 from checkpoint."""
    model = NAFNet(
        img_channel=3, width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("model", ckpt.get("state_dict", ckpt))))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    if fp16:
        model = model.half()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  NAFNet-width64: {params_m:.1f}M params, VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    return model


def denoise_video(
    input_path, output_path, checkpoint_path,
    batch_size=1, crf=18, encoder="libx265",
    max_frames=-1, fp16=True, use_compile=False,
):
    """Stream video through NAFNet with threaded IO."""

    # Load model
    model = load_nafnet(checkpoint_path, device=DEVICE, fp16=fp16, use_compile=use_compile)

    # Warmup (important for torch.compile)
    if use_compile:
        print("  Warming up compiled model...")
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE)
        if fp16:
            dummy = dummy.half()
        with torch.no_grad():
            _ = model(dummy)
        del dummy
        torch.cuda.empty_cache()

    # Video info
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    print(f"\nInput: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    # FFmpeg setup
    ffmpeg = get_ffmpeg()
    frame_bytes = w * h * 3
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    read_cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
    ]
    if max_frames > 0:
        read_cmd += ["-vframes", str(max_frames)]
    read_cmd += ["pipe:1"]

    fps_str = f"{fps:.6f}"
    if encoder == "hevc_nvenc":
        write_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-i", "pipe:0",
            "-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq",
            "-rc", "vbr", "-cq", str(crf), "-pix_fmt", "p010le",
            "-movflags", "+faststart", output_path,
        ]
    else:
        write_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-i", "pipe:0",
            "-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p10le",
            "-x265-params", "aq-mode=3:aq-strength=0.8:deblock=-1,-1:no-sao=1:rc-lookahead=40:pools=4",
            "-movflags", "+faststart", output_path,
        ]

    # Threaded IO
    frame_queue = Queue(maxsize=batch_size * 4)
    result_queue = Queue(maxsize=batch_size * 4)
    SENTINEL = object()
    WRITER_DONE = object()

    def reader_thread():
        proc = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            frame_queue.put(frame)
        frame_queue.put(SENTINEL)
        proc.stdout.close()
        proc.wait()

    writer_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=frame_bytes * 4)

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

    # Batched inference
    print(f"\nProcessing: batch_size={batch_size}, {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}")
    print(f"Encoding: {encoder} CRF {crf}")

    start = time.time()
    processed = 0
    errors = 0
    done = False

    while not done:
        # Collect batch
        batch_frames = []
        for _ in range(batch_size):
            frame = frame_queue.get()
            if frame is SENTINEL:
                done = True
                break
            batch_frames.append(frame)

        if not batch_frames:
            break

        try:
            batch_np = np.stack(batch_frames)
            batch_t = torch.from_numpy(batch_np.transpose(0, 3, 1, 2).copy()) / 255.0
            if fp16:
                batch_t = batch_t.half()
            batch_t = batch_t.to(DEVICE)

            with torch.no_grad():
                out_t = model(batch_t)
            torch.cuda.synchronize()

            out_np = (out_t.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
            del batch_t, out_t

            for i in range(len(batch_frames)):
                result_queue.put(out_np[i])

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  OOM at batch_size={len(batch_frames)}, processing one at a time")
                for frame in batch_frames:
                    try:
                        img_t = torch.from_numpy(frame.transpose(2, 0, 1).copy()).unsqueeze(0) / 255.0
                        if fp16:
                            img_t = img_t.half()
                        img_t = img_t.to(DEVICE)
                        with torch.no_grad():
                            out_t = model(img_t)
                        out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        result_queue.put(out)
                        del img_t, out_t
                    except RuntimeError:
                        result_queue.put(frame)
                        errors += 1
                        torch.cuda.empty_cache()
            else:
                for frame in batch_frames:
                    result_queue.put(frame)
                    errors += 1
                print(f"  ERROR: {e}")

        processed += len(batch_frames)
        if processed % max(batch_size * 10, 50) == 0 or processed == len(batch_frames):
            elapsed = time.time() - start
            fps_actual = processed / elapsed
            eta_min = (total_frames - processed) / max(fps_actual, 0.01) / 60
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"  [{processed}/{total_frames}] {fps_actual:.1f} fps, "
                  f"VRAM: {vram:.0f}MB, ETA: {eta_min:.1f}min, errors: {errors}")

    result_queue.put(WRITER_DONE)
    writer_t.join()
    reader_t.join()

    elapsed = time.time() - start
    out_size = os.path.getsize(output_path) / 1024**2
    print(f"\n{'=' * 60}")
    print(f"DONE: {processed} frames in {elapsed / 60:.1f} min ({processed / elapsed:.1f} fps)")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path} ({out_size:.1f} MB)")
    print(f"{'=' * 60}")

    return {
        "frames": processed,
        "elapsed_min": elapsed / 60,
        "fps": processed / elapsed,
        "errors": errors,
        "output_size_mb": out_size,
    }


def main():
    parser = argparse.ArgumentParser(description="NAFNet video denoiser (distilled)")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--checkpoint", required=True, help="NAFNet checkpoint path")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per batch (NAFNet uses less VRAM than SCUNet)")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--encoder", default="libx265",
                        choices=["hevc_nvenc", "libx265"])
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (NAFNet is compile-friendly)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_nafnet.mkv"

    denoise_video(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        crf=args.crf,
        encoder=args.encoder,
        max_frames=args.max_frames,
        fp16=not args.fp32,
        use_compile=args.compile,
    )


if __name__ == "__main__":
    main()
