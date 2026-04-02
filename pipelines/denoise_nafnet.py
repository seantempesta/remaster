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
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True
from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

DEVICE = "cuda"


def load_nafnet(checkpoint_path, device="cuda", fp16=True, use_compile=False,
                width=64, middle_blk_num=12, enc_blk_nums=None, dec_blk_nums=None):
    """Load NAFNet from checkpoint with configurable architecture."""
    enc_blk_nums = enc_blk_nums or [2, 2, 4, 8]
    dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
    model = NAFNet(
        img_channel=3, width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("model", ckpt.get("state_dict", ckpt))))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Replace LayerNorm2d with compile-friendly F.layer_norm wrapper
    if use_compile:
        model = swap_layernorm_for_compile(model)
        print("  Swapped LayerNorm2d -> LayerNorm2dCompile")

    model = model.to(device)
    if fp16:
        model = model.half()

    # channels_last for better cuDNN conv performance
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  NAFNet w={width} mid={middle_blk_num}: {params_m:.1f}M params, VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f}GB")

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
    width=64, middle_blk_num=12,
):
    """Stream video through NAFNet with threaded IO."""

    # Load model
    model = load_nafnet(checkpoint_path, device=DEVICE, fp16=fp16,
                        use_compile=use_compile, width=width,
                        middle_blk_num=middle_blk_num)

    # Video info (need dimensions before warmup)
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)

    # Warmup at actual video resolution (important for torch.compile + CUDA graphs)
    if use_compile:
        h_pad = h + (16 - h % 16) % 16
        w_pad = w + (16 - w % 16) % 16
        print(f"  Warming up compiled model at {w_pad}x{h_pad}...")
        dummy = torch.randn(batch_size, 3, h_pad, w_pad, device=DEVICE)
        if fp16:
            dummy = dummy.half()
        dummy = dummy.to(memory_format=torch.channels_last)
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
            _ = model(dummy)  # second pass for CUDA graph recording
        torch.cuda.synchronize()
        print(f"  Compile warmup: {time.time() - t0:.1f}s")
        del dummy
        torch.cuda.empty_cache()
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    print(f"\nInput: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    # FFmpeg setup for encoder (subprocess — runs in own process, no GIL contention)
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
    elif encoder == "libx264":
        write_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-threads", "0", "-i", "pipe:0",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
            "-pix_fmt", "yuv420p10le",
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

    # --- Hybrid I/O: PyAV decode (in-process, no pipe) + ffmpeg encode (subprocess) ---
    # PyAV eliminates the 64KB pipe bottleneck for decoding (biggest win).
    # Encoding stays as subprocess so ffmpeg gets its own process/CPU cores.
    import av

    frame_queue = Queue(maxsize=batch_size * 8)
    result_queue = Queue(maxsize=batch_size * 8)
    SENTINEL = object()
    WRITER_DONE = object()

    def _try_increase_pipe_buf(fd, target=8 * 1024 * 1024):
        try:
            import fcntl
            fcntl.fcntl(fd, 1031, target)  # F_SETPIPE_SZ
        except Exception:
            pass

    def reader_thread():
        container = av.open(input_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 0
        count = 0
        for frame in container.decode(stream):
            frame_queue.put(frame.to_ndarray(format='rgb24'))
            count += 1
            if max_frames > 0 and count >= max_frames:
                break
        frame_queue.put(SENTINEL)
        container.close()

    writer_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=frame_bytes * 8)
    _try_increase_pipe_buf(writer_proc.stdin.fileno())

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

    # Double-buffered inference with pinned memory + CUDA streams
    print(f"\nProcessing: batch_size={batch_size}, {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}")
    print(f"Encoding: {encoder} CRF {crf}")

    start = time.time()
    processed = 0
    errors = 0

    # Pad dimensions to match warmup shape (CUDA graphs are shape-specific!)
    h_pad = h + (16 - h % 16) % 16
    w_pad = w + (16 - w % 16) % 16

    # Pre-allocate pinned CPU buffers at PADDED dims (must match compile warmup shape)
    dtype = torch.float16 if fp16 else torch.float32
    pinned = [torch.zeros(batch_size, 3, h_pad, w_pad, dtype=dtype, pin_memory=True) for _ in range(2)]

    # Pre-allocate GPU input buffers (reuse to avoid CUDA graph invalidation from alloc/free)
    gpu_inputs = [
        torch.zeros(batch_size, 3, h_pad, w_pad, dtype=dtype, device=DEVICE).to(
            memory_format=torch.channels_last)
        for _ in range(2)
    ]

    def collect_batch():
        """Collect up to batch_size frames from reader queue."""
        frames = []
        for _ in range(batch_size):
            frame = frame_queue.get()
            if frame is SENTINEL:
                return frames, True
            frames.append(frame)
        return frames, False

    def prepare_tensor(frames, buf_idx):
        """Convert frames into pre-allocated pinned buffer at padded dims, copy to GPU."""
        buf = pinned[buf_idx]
        for i, frame in enumerate(frames):
            t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            buf[i, :, :h, :w] = t.to(dtype=dtype) / 255.0
        # Copy into pre-allocated GPU buffer (no new CUDA allocation)
        gpu_inputs[buf_idx][:len(frames)].copy_(buf[:len(frames)], non_blocking=True)
        return gpu_inputs[buf_idx][:len(frames)]

    # Per-stage timing accumulators
    t_collect = 0.0
    t_prepare = 0.0
    t_infer = 0.0
    t_postproc = 0.0
    t_write = 0.0
    n_batches = 0

    # Pre-collect first batch
    buf_idx = 0
    cur_frames, done = collect_batch()
    if cur_frames:
        cur_tensor = prepare_tensor(cur_frames, buf_idx)
        torch.cuda.synchronize()  # wait for first transfer

    while cur_frames:
        t0 = time.perf_counter()

        # GPU inference on current batch
        with torch.no_grad():
            out_t = model(cur_tensor)

        # While GPU computes, prepare next batch on CPU
        t1 = time.perf_counter()
        if not done:
            next_frames, done = collect_batch()
            next_buf_idx = 1 - buf_idx
            if next_frames:
                next_tensor = prepare_tensor(next_frames, next_buf_idx)
        else:
            next_frames = []
        t2 = time.perf_counter()

        # Sync GPU, get results — crop to original dims (undo padding)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        out_cropped = out_t[:, :, :h, :w]
        out_np = (out_cropped.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
        t4 = time.perf_counter()

        # Write entire batch to encoder at once
        result_queue.put(out_np[:len(cur_frames)])
        t5 = time.perf_counter()

        # Accumulate timings
        t_infer += t1 - t0
        t_collect += t2 - t1  # collect + prepare overlap with GPU
        t_postproc += t4 - t3  # sync + cpu transfer
        t_write += t5 - t4
        n_batches += 1

        # Don't del cur_tensor — it's a view into pre-allocated gpu_inputs (avoid CUDA graph invalidation)
        processed += len(cur_frames)

        if processed % max(batch_size * 10, 50) == 0 or processed == len(cur_frames):
            elapsed = time.time() - start
            fps_actual = processed / elapsed
            eta_min = (total_frames - processed) / max(fps_actual, 0.01) / 60
            vram = torch.cuda.max_memory_reserved() / 1024**3
            print(f"  [{processed}/{total_frames}] {fps_actual:.1f} fps, "
                  f"VRAM: {vram:.1f}GB, ETA: {eta_min:.1f}min, errors: {errors}")

        # Swap buffers
        cur_frames = next_frames
        buf_idx = 1 - buf_idx
        if cur_frames:
            cur_tensor = next_tensor

    # Print timing breakdown
    if n_batches > 0:
        print(f"\n  Timing per batch ({n_batches} batches, {batch_size} frames/batch):")
        print(f"    Infer (launch):   {t_infer/n_batches*1000:.1f}ms")
        print(f"    Collect+Prepare:  {t_collect/n_batches*1000:.1f}ms (overlaps with GPU)")
        print(f"    GPU sync+postproc:{t_postproc/n_batches*1000:.1f}ms")
        print(f"    Queue write:      {t_write/n_batches*1000:.1f}ms")
        print(f"    Total per batch:  {(t_infer+t_collect+t_postproc+t_write)/n_batches*1000:.1f}ms"
              f" = {n_batches*batch_size/(t_infer+t_collect+t_postproc+t_write):.1f} fps")

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
                        choices=["hevc_nvenc", "libx265", "libx264"])
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (NAFNet is compile-friendly)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    parser.add_argument("--width", type=int, default=64,
                        help="NAFNet channel width (default: 64)")
    parser.add_argument("--middle-blk-num", type=int, default=12,
                        help="Number of middle blocks (default: 12)")
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
        width=args.width,
        middle_blk_num=args.middle_blk_num,
    )


if __name__ == "__main__":
    main()
