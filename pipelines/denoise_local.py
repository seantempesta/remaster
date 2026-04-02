"""
NAFNet local inference with optional INT8 quantization.

Designed for RTX 3060 (6GB VRAM). Separate from the cloud pipeline
(denoise_nafnet.py / modal_denoise.py) — this is a standalone local path.

Supports:
  - fp16 baseline inference
  - TorchAO INT8 weight-only quantization (with fallback to PyTorch native)
  - torch.compile acceleration
  - Eval mode for quality comparison on validation pairs
  - Video processing with PyAV decode + ffmpeg encode

Usage:
    # Eval mode — compare fp16 vs INT8 quality
    python pipelines/denoise_local.py --eval --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize none
    python pipelines/denoise_local.py --eval --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize int8

    # Process a video
    python pipelines/denoise_local.py --input video.mkv --output denoised.mkv --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize int8

    # With torch.compile
    python pipelines/denoise_local.py --input video.mkv --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize int8 --compile
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
VAL_PAIRS_DIR = Path(__file__).resolve().parent.parent / "data" / "val_pairs"


def load_nafnet(checkpoint_path, device="cuda", fp16=True):
    """Load NAFNet-width64 from checkpoint (no compile or quantize yet)."""
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
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  NAFNet-width64: {params_m:.1f}M params, VRAM after load: {vram_mb:.0f}MB")

    return model


def apply_quantization(model, quantize_mode="int8"):
    """Apply INT8 quantization. Try TorchAO first, fall back to PyTorch native."""
    if quantize_mode == "none":
        print("  Quantization: none (fp16 weights)")
        return model

    vram_before = torch.cuda.memory_allocated() / 1024**2

    if quantize_mode == "int8":
        # Try TorchAO first — API varies by version
        try:
            from torchao.quantization import quantize_
            # Try newer API first (torchao >= 0.8)
            try:
                from torchao.quantization import Int8WeightOnlyConfig
                quantize_(model, Int8WeightOnlyConfig())
            except ImportError:
                # Older API (torchao 0.5-0.7)
                from torchao.quantization import int8_weight_only
                quantize_(model, int8_weight_only())
            vram_after = torch.cuda.memory_allocated() / 1024**2
            print(f"  Quantization: TorchAO INT8 weight-only")
            print(f"  VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB (saved {vram_before - vram_after:.0f}MB)")
            return model
        except ImportError:
            print("  TorchAO not installed, falling back to PyTorch native quantization")
        except Exception as e:
            print(f"  TorchAO quantization failed ({e}), falling back to PyTorch native")

        # Fallback: PyTorch native dynamic quantization (CPU only)
        try:
            model_cpu = model.float().cpu()
            model_q = torch.quantization.quantize_dynamic(
                model_cpu, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
            )
            vram_after = torch.cuda.memory_allocated() / 1024**2
            print(f"  Quantization: PyTorch native dynamic INT8 (CPU-only fallback)")
            print(f"  WARNING: Native dynamic quantization runs on CPU, will be very slow")
            # Tag model so callers know it's on CPU
            model_q._on_cpu = True
            return model_q
        except Exception as e:
            print(f"  PyTorch native quantization also failed: {e}")
            print(f"  Continuing with fp16 weights")
            return model

    print(f"  Unknown quantize mode '{quantize_mode}', using fp16")
    return model


def prepare_model(checkpoint_path, quantize="none", use_compile=False, fp16=True):
    """Load, quantize, compile, and warmup the model."""
    model = load_nafnet(checkpoint_path, device=DEVICE, fp16=fp16)

    # Apply quantization before compile (TorchAO recommends this order)
    model = apply_quantization(model, quantize)

    # channels_last for better cuDNN performance
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    # Swap LayerNorm for compile-friendly version
    if use_compile:
        model = swap_layernorm_for_compile(model)
        print("  Swapped LayerNorm2d -> LayerNorm2dCompile")

    # torch.compile — try reduce-overhead, fall back to default, then to eager
    if use_compile:
        compiled = False
        for mode in ["reduce-overhead", "default"]:
            try:
                model_compiled = torch.compile(model, mode=mode)
                # Test with a small dummy to catch Triton/backend errors early
                dummy = torch.randn(1, 3, 64, 64, dtype=torch.float16 if fp16 else torch.float32,
                                    device=DEVICE).to(memory_format=torch.channels_last)
                with torch.no_grad():
                    _ = model_compiled(dummy)
                del dummy
                torch.cuda.empty_cache()
                model = model_compiled
                print(f"  torch.compile enabled ({mode} mode)")
                compiled = True
                break
            except Exception as e:
                print(f"  torch.compile mode='{mode}' failed: {e}")
        if not compiled:
            print("  torch.compile unavailable (Triton not installed on Windows), using eager mode")

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak VRAM after setup: {peak_vram:.0f}MB")
    return model


def run_eval(model, max_frames=-1, fp16=True):
    """Run model on validation pairs and report metrics."""
    input_dir = VAL_PAIRS_DIR / "input"
    target_dir = VAL_PAIRS_DIR / "target"

    if not input_dir.exists():
        print(f"ERROR: Validation pairs not found at {VAL_PAIRS_DIR}")
        return

    from PIL import Image

    input_files = sorted(input_dir.glob("*.png"))
    if max_frames > 0:
        input_files = input_files[:max_frames]

    print(f"\nEval: {len(input_files)} frames from {VAL_PAIRS_DIR}")

    torch.cuda.reset_peak_memory_stats()
    total_psnr = 0.0
    total_l1 = 0.0
    total_time = 0.0
    count = 0

    dtype = torch.float16 if fp16 else torch.float32

    for img_path in input_files:
        target_path = target_dir / img_path.name
        if not target_path.exists():
            print(f"  Skipping {img_path.name} (no target)")
            continue

        # Load input
        inp_np = np.array(Image.open(str(img_path)).convert("RGB"))
        inp_t = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
        inp_t = inp_t.to(DEVICE, memory_format=torch.channels_last)

        # Pad to multiple of 16
        _, _, h, w = inp_t.shape
        h_pad = h + (16 - h % 16) % 16
        w_pad = w + (16 - w % 16) % 16
        if h_pad != h or w_pad != w:
            inp_padded = torch.zeros(1, 3, h_pad, w_pad, dtype=dtype, device=DEVICE)
            inp_padded = inp_padded.to(memory_format=torch.channels_last)
            inp_padded[:, :, :h, :w] = inp_t
            inp_t = inp_padded

        # Inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_t = model(inp_t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Crop back to original size
        out_t = out_t[:, :, :h, :w].clamp(0, 1)
        total_time += t1 - t0

        # Load target and compute metrics
        tgt_np = np.array(Image.open(str(target_path)).convert("RGB"))
        tgt_t = torch.from_numpy(tgt_np).permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
        tgt_t = tgt_t.to(DEVICE)

        # PSNR
        mse = torch.mean((out_t - tgt_t) ** 2).item()
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = 100.0
        total_psnr += psnr

        # L1
        l1 = torch.mean(torch.abs(out_t - tgt_t)).item()
        total_l1 += l1

        count += 1
        print(f"  [{count}/{len(input_files)}] {img_path.name}: PSNR={psnr:.2f}dB, L1={l1:.4f}, "
              f"time={t1 - t0:.3f}s")

        # Free GPU memory
        del inp_t, out_t, tgt_t
        if h_pad != h or w_pad != w:
            del inp_padded

    if count == 0:
        print("No frames processed!")
        return

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    avg_psnr = total_psnr / count
    avg_l1 = total_l1 / count
    avg_fps = count / total_time

    print(f"\n{'=' * 60}")
    print(f"EVAL RESULTS ({count} frames)")
    print(f"  Avg PSNR:   {avg_psnr:.2f} dB")
    print(f"  Avg L1:     {avg_l1:.4f}")
    print(f"  Avg FPS:    {avg_fps:.1f}")
    print(f"  Peak VRAM:  {peak_vram:.0f} MB")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    return {
        "avg_psnr": avg_psnr,
        "avg_l1": avg_l1,
        "avg_fps": avg_fps,
        "peak_vram_mb": peak_vram,
        "frames": count,
    }


def denoise_video(
    input_path, output_path, model,
    batch_size=1, crf=18, encoder="libx265",
    max_frames=-1, fp16=True,
):
    """Stream video through NAFNet with threaded IO (local version)."""
    import av

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    print(f"\nInput: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    # FFmpeg encoder setup — local only uses libx265 or libx264
    ffmpeg_bin = get_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_bytes = w * h * 3
    fps_str = f"{fps:.6f}"

    if encoder == "libx264":
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

    # Threaded IO
    frame_queue = Queue(maxsize=batch_size * 8)
    result_queue = Queue(maxsize=batch_size * 8)
    SENTINEL = object()
    WRITER_DONE = object()

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

    print(f"\nProcessing: batch_size={batch_size}, {'fp16' if fp16 else 'fp32'}")
    print(f"Encoding: {encoder} CRF {crf}")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    processed = 0
    dtype = torch.float16 if fp16 else torch.float32

    # Pad dimensions to multiple of 16
    h_pad = h + (16 - h % 16) % 16
    w_pad = w + (16 - w % 16) % 16

    while True:
        # Collect batch
        frames = []
        done = False
        for _ in range(batch_size):
            frame = frame_queue.get()
            if frame is SENTINEL:
                done = True
                break
            frames.append(frame)

        if not frames:
            break

        # Prepare tensor
        batch_t = torch.zeros(len(frames), 3, h_pad, w_pad, dtype=dtype, device=DEVICE)
        batch_t = batch_t.to(memory_format=torch.channels_last)
        for i, f in enumerate(frames):
            t = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).to(dtype) / 255.0
            batch_t[i, :, :h, :w] = t

        # Inference
        with torch.no_grad():
            out_t = model(batch_t)

        # Crop and convert
        out_cropped = out_t[:, :, :h, :w]
        out_np = (out_cropped.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
        result_queue.put(out_np[:len(frames)])

        processed += len(frames)
        if processed % max(batch_size * 10, 50) == 0 or processed == len(frames):
            elapsed = time.time() - start
            fps_actual = processed / elapsed
            eta_min = (total_frames - processed) / max(fps_actual, 0.01) / 60
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"  [{processed}/{total_frames}] {fps_actual:.1f} fps, "
                  f"VRAM: {vram:.0f}MB, ETA: {eta_min:.1f}min")

        del batch_t, out_t
        if done:
            break

    result_queue.put(WRITER_DONE)
    writer_t.join()
    reader_t.join()

    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    out_size = os.path.getsize(output_path) / 1024**2

    print(f"\n{'=' * 60}")
    print(f"DONE: {processed} frames in {elapsed / 60:.1f} min ({processed / elapsed:.1f} fps)")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")
    print(f"  Output: {output_path} ({out_size:.1f} MB)")
    print(f"{'=' * 60}")

    return {
        "frames": processed,
        "elapsed_min": elapsed / 60,
        "fps": processed / elapsed,
        "peak_vram_mb": peak_vram,
        "output_size_mb": out_size,
    }


def main():
    parser = argparse.ArgumentParser(description="NAFNet local inference with INT8 quantization")
    parser.add_argument("--input", default=None, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--checkpoint", required=True, help="NAFNet checkpoint path")
    parser.add_argument("--quantize", default="none", choices=["none", "int8"],
                        help="Quantization mode (default: none = fp16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per batch (default: 1)")
    parser.add_argument("--crf", type=int, default=18, help="CRF for encoding")
    parser.add_argument("--encoder", default="libx265", choices=["libx265", "libx264"],
                        help="Video encoder (local only, no NVENC)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--eval", action="store_true", help="Run on val_pairs and report metrics")
    args = parser.parse_args()

    if not args.eval and not args.input:
        parser.error("Either --eval or --input is required")

    fp16 = True
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"Quantize: {args.quantize}, Compile: {args.compile}, fp16: {fp16}")
    print()

    model = prepare_model(
        args.checkpoint,
        quantize=args.quantize,
        use_compile=args.compile,
        fp16=fp16,
    )

    if args.eval:
        run_eval(model, max_frames=args.max_frames, fp16=fp16)
    else:
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_local.mkv"

        denoise_video(
            input_path=args.input,
            output_path=args.output,
            model=model,
            batch_size=args.batch_size,
            crf=args.crf,
            encoder=args.encoder,
            max_frames=args.max_frames,
            fp16=fp16,
        )


if __name__ == "__main__":
    main()
