"""
Unified video remaster pipeline -- DRUNet teacher/student inference.

Streaming architecture: PyAV decode -> torch inference -> FFmpeg encode.
Double-buffered pinned memory transfers, CUDA event timing, optional torch.compile.

Usage:
    # DRUNet teacher (32.6M params, ~5 fps on RTX 3060)
    python pipelines/remaster.py -i data/clip_mid_1080p.mp4 \
        -c checkpoints/drunet_teacher/best.pth --nc-list 64,128,256,512 --nb 4

    # DRUNet student (1.06M params, ~30 fps FP16)
    python pipelines/remaster.py -i data/clip_mid_1080p.mp4 \
        -c checkpoints/drunet_student/best.pth --nc-list 16,32,64,128 --nb 2 --compile

    # Hardware encode + audio passthrough
    python pipelines/remaster.py -i episode.mkv -c checkpoint.pth \
        --encoder hevc_nvenc --mux-audio

    # Quick test (first 100 frames)
    python pipelines/remaster.py -i video.mp4 -c checkpoint.pth --max-frames 100
"""
import sys
import os
import time
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._inductor.config

# Inductor optimizations for Conv+ReLU UNet inference
torch._inductor.config.conv_1x1_as_mm = True           # 1x1 conv as matmul (Tensor Core)
torch._inductor.config.freezing = True                  # Inline weights as constants
torch._inductor.config.freezing_discard_parameters = True  # Free param memory after freeze
torch._inductor.config.coordinate_descent_tuning = True  # Better Triton kernel params

from lib.paths import add_kair_to_path
from lib.ffmpeg_utils import get_ffmpeg, get_video_info, build_encoder_cmd

DEVICE = "cuda"
PAD_MULTIPLE = 8  # DRUNet uses 3 levels of 2x downsample -> must be divisible by 8


def load_model(checkpoint_path, nc_list, nb, device="cuda", fp16=True, use_compile=False):
    """Load DRUNet (UNetRes) from checkpoint."""
    add_kair_to_path()
    from models.network_unet import UNetRes

    # Auto-detect in_nc from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt)))

    # Check first conv weight shape to detect in_nc
    head_key = "m_head.0.weight"
    if head_key in state_dict:
        in_nc = state_dict[head_key].shape[1]
    else:
        in_nc = 3

    model = UNetRes(in_nc=in_nc, out_nc=3, nc=nc_list, nb=nb,
                    act_mode='R', bias=False)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}...")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    if fp16:
        model = model.half()
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    iteration = ckpt.get("iteration", "?")
    psnr = ckpt.get("psnr", "?")
    print(f"  DRUNet nc={nc_list} nb={nb} in_nc={in_nc}: {params_m:.1f}M params")
    print(f"  Checkpoint: iter={iteration}, psnr={psnr}")
    print(f"  VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f}GB")

    # Free checkpoint from RAM
    del ckpt, state_dict

    if use_compile:
        try:
            # reduce-overhead uses CUDA graphs for minimal CPU dispatch overhead
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            print("  torch.compile enabled (reduce-overhead, fullgraph, freezing)")
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    return model


def remaster_video(
    model, input_path, output_path, *,
    batch_size=1, crf=18, encoder="hevc_nvenc",
    max_frames=-1, fp16=True, use_compile=False,
    hwdec=False, mux_audio=False,
):
    """Stream video through DRUNet with threaded IO and double-buffered inference."""
    import av

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)

    # Pad to PAD_MULTIPLE
    h_pad = h + (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    w_pad = w + (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE

    # Warmup for torch.compile (CUDA graph recording needs exact shape)
    if use_compile:
        print(f"  Warming up compiled model at {w_pad}x{h_pad}...")
        dtype = torch.float16 if fp16 else torch.float32
        dummy = torch.randn(batch_size, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)
        dummy = dummy.to(memory_format=torch.channels_last)
        t0 = time.time()
        with torch.inference_mode():
            _ = model(dummy)
            _ = model(dummy)  # second pass for CUDA graph recording
            _ = model(dummy)  # third pass to ensure graph is fully recorded
        torch.cuda.synchronize()
        print(f"  Compile warmup: {time.time() - t0:.1f}s")
        del dummy
        torch.cuda.empty_cache()

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print(f"\nInput: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")

    # If muxing audio, encode video to temp file first
    if mux_audio:
        video_output = tempfile.mktemp(suffix=".mkv", prefix="remaster_video_")
    else:
        video_output = output_path

    print(f"Output: {output_path}")

    # Encoder command
    ffmpeg_bin = get_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_cmd = build_encoder_cmd(ffmpeg_bin, w, h, fps, video_output, encoder, crf)
    frame_bytes = w * h * 3

    # Threading setup — larger queues to decouple stages
    frame_queue = Queue(maxsize=batch_size * 16)
    result_queue = Queue(maxsize=batch_size * 16)
    SENTINEL = object()
    WRITER_DONE = object()

    def reader_thread_pyav():
        """Decode with PyAV (CPU, threaded -- eliminates pipe bottleneck)."""
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

    def reader_thread_hwdec():
        """Decode with FFmpeg NVDEC via pipe."""
        dec_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-c:v", "hevc_cuvid",
            "-i", input_path,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-v", "quiet", "pipe:1",
        ]
        proc = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
        count = 0
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame_queue.put(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
            count += 1
            if max_frames > 0 and count >= max_frames:
                break
        frame_queue.put(SENTINEL)
        proc.stdout.close()
        proc.wait()

    writer_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=frame_bytes * 16)

    def writer_thread():
        while True:
            item = result_queue.get()
            if item is WRITER_DONE:
                break
            writer_proc.stdin.write(item.tobytes())
        writer_proc.stdin.close()
        writer_proc.wait()

    reader_fn = reader_thread_hwdec if hwdec else reader_thread_pyav
    reader_t = Thread(target=reader_fn, daemon=True)
    writer_t = Thread(target=writer_thread, daemon=True)
    reader_t.start()
    writer_t.start()

    # Double-buffered inference
    print(f"\nProcessing: batch_size={batch_size}, {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}")
    print(f"Encoding: {encoder} CRF {crf}")

    dtype = torch.float16 if fp16 else torch.float32
    pinned = [torch.zeros(batch_size, 3, h_pad, w_pad, dtype=dtype, pin_memory=True) for _ in range(2)]
    gpu_inputs = [
        torch.zeros(batch_size, 3, h_pad, w_pad, dtype=dtype, device=DEVICE).to(
            memory_format=torch.channels_last)
        for _ in range(2)
    ]

    # CUDA events for accurate GPU timing
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    gpu_ms_total = 0.0

    start = time.time()
    processed = 0
    n_batches = 0
    t_infer = t_collect = t_postproc = t_write = 0.0

    def collect_batch():
        frames = []
        for _ in range(batch_size):
            frame = frame_queue.get()
            if frame is SENTINEL:
                return frames, True
            frames.append(frame)
        return frames, False

    def prepare_tensor(frames, buf_idx):
        buf = pinned[buf_idx]
        for i, frame in enumerate(frames):
            t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            buf[i, :, :h, :w] = t.to(dtype=dtype) / 255.0
        gpu_inputs[buf_idx][:len(frames)].copy_(buf[:len(frames)], non_blocking=True)
        return gpu_inputs[buf_idx][:len(frames)]

    # Pre-collect first batch
    buf_idx = 0
    cur_frames, done = collect_batch()
    if cur_frames:
        cur_tensor = prepare_tensor(cur_frames, buf_idx)
        torch.cuda.synchronize()

    while cur_frames:
        t0 = time.perf_counter()

        # GPU inference with CUDA event timing
        ev_start.record()
        with torch.inference_mode():
            out_t = model(cur_tensor)
        ev_end.record()

        # Overlap: prepare next batch on CPU while GPU computes
        t1 = time.perf_counter()
        if not done:
            next_frames, done = collect_batch()
            next_buf_idx = 1 - buf_idx
            if next_frames:
                next_tensor = prepare_tensor(next_frames, next_buf_idx)
        else:
            next_frames = []
        t2 = time.perf_counter()

        # Sync GPU, get results
        torch.cuda.synchronize()
        gpu_ms_total += ev_start.elapsed_time(ev_end)
        t3 = time.perf_counter()

        out_cropped = out_t[:, :, :h, :w]
        out_np = (out_cropped.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)

        # Color is correct — VLC had a display bug, not a model bug.
        # BT.709 metadata is set in build_encoder_cmd (lib/ffmpeg_utils.py).

        t4 = time.perf_counter()

        result_queue.put(out_np[:len(cur_frames)])
        t5 = time.perf_counter()

        t_infer += t1 - t0
        t_collect += t2 - t1
        t_postproc += t4 - t3
        t_write += t5 - t4
        n_batches += 1

        processed += len(cur_frames)

        if processed % max(batch_size * 10, 50) == 0 or processed == len(cur_frames):
            elapsed = time.time() - start
            fps_wall = processed / elapsed
            fps_gpu = processed / (gpu_ms_total / 1000) if gpu_ms_total > 0 else 0
            vram = torch.cuda.max_memory_reserved() / 1024**3
            eta_min = (total_frames - processed) / max(fps_wall, 0.01) / 60
            print(f"  [{processed}/{total_frames}] wall={fps_wall:.1f}fps gpu={fps_gpu:.1f}fps "
                  f"VRAM={vram:.1f}GB ETA={eta_min:.1f}min")

        # Swap buffers
        cur_frames = next_frames
        buf_idx = 1 - buf_idx
        if cur_frames:
            cur_tensor = next_tensor

    # Timing breakdown
    if n_batches > 0:
        print(f"\nTiming per batch ({n_batches} batches, {batch_size} frames/batch):")
        print(f"  Infer (launch):   {t_infer/n_batches*1000:.1f}ms")
        print(f"  Collect+Prepare:  {t_collect/n_batches*1000:.1f}ms (overlaps GPU)")
        print(f"  GPU sync+postproc:{t_postproc/n_batches*1000:.1f}ms")
        print(f"  Queue write:      {t_write/n_batches*1000:.1f}ms")
        total_batch_ms = (t_infer + t_collect + t_postproc + t_write) / n_batches * 1000
        print(f"  Total per batch:  {total_batch_ms:.1f}ms"
              f" = {n_batches * batch_size / (t_infer + t_collect + t_postproc + t_write):.1f}fps")
        print(f"  GPU-only:         {gpu_ms_total/n_batches:.1f}ms/batch"
              f" = {processed/(gpu_ms_total/1000):.1f}fps")

    result_queue.put(WRITER_DONE)
    writer_t.join()
    reader_t.join()

    # Audio mux pass
    if mux_audio and os.path.exists(video_output):
        print(f"\nMuxing audio from {input_path}...")
        mux_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_output,
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c", "copy",
            output_path,
        ]
        subprocess.run(mux_cmd, check=True)
        os.remove(video_output)

    elapsed = time.time() - start
    out_size = os.path.getsize(output_path) / 1024**2
    fps_final = processed / elapsed
    fps_gpu_final = processed / (gpu_ms_total / 1000) if gpu_ms_total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"DONE: {processed} frames in {elapsed / 60:.1f}min")
    print(f"  Wall: {fps_final:.1f} fps")
    print(f"  GPU:  {fps_gpu_final:.1f} fps ({gpu_ms_total/processed:.1f}ms/frame)")
    print(f"  Output: {output_path} ({out_size:.1f}MB)")
    print(f"{'=' * 60}")

    return {
        "frames": processed,
        "elapsed_min": elapsed / 60,
        "fps_wall": fps_final,
        "fps_gpu": fps_gpu_final,
        "gpu_ms_per_frame": gpu_ms_total / max(processed, 1),
        "output_size_mb": out_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Video remaster pipeline (DRUNet)")

    # Required
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")

    # Model architecture
    parser.add_argument("--nc-list", default="64,128,256,512",
                        help="Channel counts per UNet level (default: 64,128,256,512 for teacher)")
    parser.add_argument("--nb", type=int, default=4,
                        help="Residual blocks per level (default: 4 for teacher)")

    # Output
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--encoder", default="hevc_nvenc",
                        choices=["hevc_nvenc", "libx265", "libx264"])
    parser.add_argument("--crf", type=int, default=18)

    # Performance
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (reduce-overhead mode)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)

    # Decode/encode options
    parser.add_argument("--hwdec", action="store_true",
                        help="Use NVDEC hardware decode (HEVC only)")
    parser.add_argument("--mux-audio", action="store_true",
                        help="Copy audio from input to output")
    # --preserve-color removed: VLC display bug was the cause, not the model

    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_remaster.mkv")

    nc_list = [int(x) for x in args.nc_list.split(",")]
    fp16 = not args.fp32

    print(f"Loading model: {args.checkpoint}")
    model = load_model(
        args.checkpoint, nc_list=nc_list, nb=args.nb,
        device=DEVICE, fp16=fp16, use_compile=args.compile,
    )

    remaster_video(
        model, args.input, args.output,
        batch_size=args.batch_size, crf=args.crf, encoder=args.encoder,
        max_frames=args.max_frames, fp16=fp16, use_compile=args.compile,
        hwdec=args.hwdec, mux_audio=args.mux_audio,
    )


if __name__ == "__main__":
    main()
