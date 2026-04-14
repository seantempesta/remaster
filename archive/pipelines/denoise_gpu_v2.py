"""
Zero-copy GPU pipeline v2 — CUDA streams + ring buffers for true hardware overlap.

NVDEC, CUDA inference, and NVENC are physically separate hardware units on the GPU.
This pipeline uses separate CUDA streams so they run concurrently on different frames:

    Stream 0 (decode):  NVDEC decode → copy to ring buffer
    Stream 1 (infer):   normalize → NAFNet → denormalize → ARGB
    Stream 2 (encode):  NVENC encode → write bitstream

CUDA events synchronize between stages without touching Python or the GIL.
Triple-buffered: each stage works on a different frame simultaneously.

Usage:
    python pipelines/denoise_gpu_v2.py --input episode.mkv --checkpoint checkpoints/nafnet_w32_mid4/nafnet_best.pth --width 32 --middle-blk-num 4 --compile
    python pipelines/denoise_gpu_v2.py --input episode.mkv --checkpoint ckpt.pth --max-frames 100
"""
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True

# PyNvVideoCodec needs CUDA runtime DLLs
if sys.platform == "win32":
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

DEVICE = "cuda"
N_SLOTS = 3  # triple buffering


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
    del ckpt
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if use_compile:
        model = swap_layernorm_for_compile(model)
        print("  Swapped LayerNorm2d -> LayerNorm2dCompile")

    model = model.to(device)
    if fp16:
        model = model.half()

    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  NAFNet w={width} mid={middle_blk_num}: {params_m:.1f}M params, "
          f"VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f}GB")

    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    return model


def wrap_bitstream(raw_path, output_path, input_path, fps, codec):
    """Wrap raw bitstream in a container and mux audio/subs from original."""
    ffmpeg = get_ffmpeg()
    raw_codec = "hevc" if codec == "hevc" else "h264"

    wrapped_video = raw_path + ".mp4"
    cmd_wrap = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-fflags", "+genpts",
        "-f", raw_codec, "-r", f"{fps:.6f}", "-i", raw_path,
        "-c:v", "copy",
        wrapped_video,
    ]
    print(f"\nWrapping bitstream...")
    r = subprocess.run(cmd_wrap, capture_output=True, text=True)
    if r.returncode != 0:
        # Fallback: try with explicit timestamp generation
        cmd_wrap2 = [
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            "-fflags", "+genpts+igndts",
            "-f", raw_codec, "-r", f"{fps:.6f}", "-i", raw_path,
            "-c:v", "copy", "-vsync", "cfr",
            wrapped_video,
        ]
        r = subprocess.run(cmd_wrap2, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  Wrap failed: {r.stderr.strip()}")
            return False

    cmd_mux = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", wrapped_video,
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"Muxing audio from original...")
    r = subprocess.run(cmd_mux, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  Audio mux warning: {r.stderr.strip()}")
        os.replace(wrapped_video, output_path)
        print("  Video only (no audio).")
        return True

    for f in [wrapped_video]:
        if os.path.exists(f):
            os.remove(f)
    print(f"  Muxed output: {output_path}")
    return True


def denoise_video(
    input_path, output_path, checkpoint_path,
    crf=18, codec="hevc", max_frames=-1, fp16=True,
    use_compile=False, width=64, middle_blk_num=12,
):
    """CUDA-streams GPU pipeline: NVDEC -> NAFNet -> NVENC with true overlap."""

    try:
        import PyNvVideoCodec as nvc
    except ImportError as e:
        print(f"ERROR: PyNvVideoCodec not available: {e}")
        print("  Install with: pip install PyNvVideoCodec")
        sys.exit(1)

    # Load model
    model = load_nafnet(checkpoint_path, device=DEVICE, fp16=fp16,
                        use_compile=use_compile, width=width,
                        middle_blk_num=middle_blk_num)

    # Get video info
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    _, _, fps_info, total_frames, duration = get_video_info(input_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # --- Create CUDA streams FIRST ---
    # Each stream targets different hardware: NVDEC, CUDA cores, NVENC
    # CRITICAL: Pass stream handles to PyNvVideoCodec so their internal
    # cudaStreamSynchronize() only blocks their own stream, not inference.
    decode_stream = torch.cuda.Stream()
    infer_stream = torch.cuda.Stream()
    encode_stream = torch.cuda.Stream()
    print(f"\n  Created 3 CUDA streams for decode/infer/encode overlap")
    print(f"    decode_stream={decode_stream.cuda_stream}, "
          f"infer_stream={infer_stream.cuda_stream}, "
          f"encode_stream={encode_stream.cuda_stream}")

    # --- NVDEC Decoder ---
    # Pass decode_stream so NVDEC's internal cuStreamSynchronize only
    # blocks decode_stream, not the default stream (which would block inference).
    print(f"Setting up NVDEC decoder (stream={decode_stream.cuda_stream})...")
    decoder = nvc.SimpleDecoder(
        input_path, gpu_id=0,
        cuda_stream=decode_stream.cuda_stream,
        output_color_type=nvc.OutputColorType.RGBP,
    )

    frame_iter = iter(decoder)
    try:
        first_raw = next(frame_iter)
    except StopIteration:
        print("ERROR: Could not decode any frames from input.")
        sys.exit(1)

    first_frame = torch.from_dlpack(first_raw)  # [3, H, W] uint8 on cuda
    _, h, w = first_frame.shape
    print(f"  Decoded first frame: {w}x{h}")

    h_pad = h + (16 - h % 16) % 16
    w_pad = w + (16 - w % 16) % 16
    dtype = torch.float16 if fp16 else torch.float32

    # --- Pre-allocate ring buffers (triple buffered) ---
    # Decoded frames: [3, H, W] uint8
    decoded_bufs = [torch.empty(3, h, w, dtype=torch.uint8, device=DEVICE) for _ in range(N_SLOTS)]
    # Model input: [1, 3, H_pad, W_pad] fp16, channels_last
    input_bufs = [
        torch.zeros(1, 3, h_pad, w_pad, dtype=dtype, device=DEVICE).to(memory_format=torch.channels_last)
        for _ in range(N_SLOTS)
    ]
    # ARGB output for encoder: [H, W, 4] uint8
    argb_bufs = [torch.empty(h, w, 4, dtype=torch.uint8, device=DEVICE) for _ in range(N_SLOTS)]
    for buf in argb_bufs:
        buf[:, :, 0] = 255  # alpha channel

    # CUDA events for inter-stage synchronization
    decode_done = [torch.cuda.Event() for _ in range(N_SLOTS)]
    infer_done = [torch.cuda.Event() for _ in range(N_SLOTS)]

    vram_after_alloc = torch.cuda.max_memory_reserved() / 1024**3
    print(f"  Ring buffers allocated ({N_SLOTS} slots), VRAM: {vram_after_alloc:.1f}GB")

    # --- torch.compile warmup ---
    # CRITICAL: warmup must run on infer_stream so CUDA graphs are captured there.
    # If captured on default stream, graph.replay() always runs on stream 0
    # regardless of the active stream context, serializing with decode/encode.
    if use_compile:
        print(f"  Warming up compiled model on infer_stream at {w_pad}x{h_pad}...")
        with torch.cuda.stream(infer_stream):
            dummy = torch.randn(1, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)
            dummy = dummy.to(memory_format=torch.channels_last)
            t0 = time.time()
            with torch.no_grad():
                _ = model(dummy)
                _ = model(dummy)
                _ = model(dummy)  # third pass to ensure graph is fully recorded
        infer_stream.synchronize()
        print(f"  Compile warmup: {time.time() - t0:.1f}s")
        del dummy
        torch.cuda.empty_cache()

    # --- NVENC Encoder ---
    print(f"Setting up NVENC encoder...")
    raw_path = output_path + ".raw"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Pass encode_stream so NVENC's internal nvEncLockBitstream only
    # blocks encode_stream, not the default stream.
    encoder = nvc.CreateEncoder(
        w, h, "ARGB", False,
        codec=codec,
        preset="P4",
        cudastream=encode_stream.cuda_stream,
    )
    print(f"  NVENC encoder: {codec} ARGB, {w}x{h}")

    out_file = open(raw_path, "wb")

    # --- Pipeline state ---
    SENTINEL = object()
    # Slot availability: which ring buffer slots are free for decode
    # We use queues to coordinate slot usage between threads
    free_slots = Queue()
    for i in range(N_SLOTS):
        free_slots.put(i)
    decoded_slots = Queue(maxsize=N_SLOTS)  # slots ready for inference
    encoded_slots = Queue(maxsize=N_SLOTS)  # slots ready for encoding

    print(f"\nProcessing: {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}, codec={codec}, "
          f"CUDA streams, {N_SLOTS}-slot ring buffer")
    print(f"Input: {input_path}")
    print(f"  {w}x{h} @ {fps_info:.3f}fps, ~{total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    start = time.time()
    errors = [0]

    # Per-stage timing accumulators (wall clock, includes waits)
    t_decode_total = [0.0]
    t_infer_total = [0.0]
    t_encode_total = [0.0]
    n_decoded = [0]
    n_inferred = [0]
    n_encoded = [0]

    # --- Decode thread ---
    def decode_thread_fn():
        count = 0
        # First frame — copy into first available slot
        slot = free_slots.get()
        t0 = time.perf_counter()
        with torch.cuda.stream(decode_stream):
            decoded_bufs[slot].copy_(first_frame, non_blocking=True)
            decode_done[slot].record(decode_stream)
        t_decode_total[0] += time.perf_counter() - t0
        decoded_slots.put(slot)
        count += 1
        n_decoded[0] += 1

        for raw_frame in frame_iter:
            if 0 < max_frames <= count:
                break
            slot = free_slots.get()
            t0 = time.perf_counter()
            with torch.cuda.stream(decode_stream):
                frame_tensor = torch.from_dlpack(raw_frame)
                decoded_bufs[slot].copy_(frame_tensor, non_blocking=True)
                decode_done[slot].record(decode_stream)
            t_decode_total[0] += time.perf_counter() - t0
            decoded_slots.put(slot)
            count += 1
            n_decoded[0] += 1

        decoded_slots.put(SENTINEL)

    # --- Inference thread ---
    def infer_thread_fn():
        while True:
            item = decoded_slots.get()
            if item is SENTINEL:
                encoded_slots.put(SENTINEL)
                break
            slot = item
            try:
                t0 = time.perf_counter()
                with torch.cuda.stream(infer_stream):
                    # Wait for decode to finish writing this slot
                    infer_stream.wait_event(decode_done[slot])

                    # Normalize uint8 -> fp16 [0,1] into padded input buffer
                    input_bufs[slot][0, :, :h, :w] = decoded_bufs[slot].to(dtype=dtype) / 255.0

                    with torch.no_grad():
                        out = model(input_bufs[slot])

                    # Crop, clamp, quantize, scatter into ARGB
                    out_rgb = (out[0, :, :h, :w].clamp(0, 1) * 255).byte()
                    argb_bufs[slot][:, :, 1] = out_rgb[0]  # R
                    argb_bufs[slot][:, :, 2] = out_rgb[1]  # G
                    argb_bufs[slot][:, :, 3] = out_rgb[2]  # B

                # Record on DEFAULT stream — torch.compile reduce-overhead
                # replays CUDA graphs on default stream regardless of the
                # torch.cuda.stream() context. Recording on infer_stream
                # would fire immediately since no work is pending there.
                infer_done[slot].record(torch.cuda.default_stream())
                t_infer_total[0] += time.perf_counter() - t0
                n_inferred[0] += 1
                encoded_slots.put(slot)
            except Exception as e:
                print(f"  Inference error: {e}")
                import traceback; traceback.print_exc()
                errors[0] += 1
                free_slots.put(slot)

    # --- Encode thread ---
    def encode_thread_fn():
        processed = 0
        while True:
            item = encoded_slots.get()
            if item is SENTINEL:
                break
            slot = item
            try:
                t0 = time.perf_counter()
                # Tell encode_stream to wait for inference completion
                encode_stream.wait_event(infer_done[slot])
                t_wait = time.perf_counter()
                # DO NOT synchronize here — let NVENC's internal sync on
                # encode_stream handle it. encoder.Encode() runs on the
                # stream we passed at construction (encode_stream).
                encoded = encoder.Encode(argb_bufs[slot])
                t_enc = time.perf_counter()
                if encoded and len(encoded) > 0:
                    out_file.write(encoded)
                t_write = time.perf_counter()

                t_encode_total[0] += t_write - t0
                # Track breakdown for first 10 frames
                if processed < 10:
                    print(f"    enc[{processed}]: wait={1000*(t_wait-t0):.1f}ms "
                          f"encode={1000*(t_enc-t_wait):.1f}ms "
                          f"write={1000*(t_write-t_enc):.1f}ms")
                n_encoded[0] += 1
                processed += 1
                # Return slot to the free pool
                free_slots.put(slot)

                if processed % 50 == 0 or processed == 1:
                    elapsed = time.time() - start
                    fps_actual = processed / elapsed
                    vram = torch.cuda.max_memory_reserved() / 1024**3
                    eta_min = ((total_frames - processed)
                               / max(fps_actual, 0.01) / 60)
                    print(f"  [{processed}/{total_frames}] "
                          f"{fps_actual:.1f} fps, "
                          f"VRAM: {vram:.1f}GB, "
                          f"ETA: {eta_min:.1f}min")
            except Exception as e:
                print(f"  Encode error: {e}")
                import traceback; traceback.print_exc()
                errors[0] += 1
                free_slots.put(slot)
                processed += 1

    # Launch pipeline
    t_dec = Thread(target=decode_thread_fn, name="decode", daemon=True)
    t_inf = Thread(target=infer_thread_fn, name="infer", daemon=True)
    t_enc = Thread(target=encode_thread_fn, name="encode", daemon=True)
    t_dec.start()
    t_inf.start()
    t_enc.start()

    t_dec.join()
    t_inf.join()
    t_enc.join()

    # Flush encoder
    flush_bytes = encoder.EndEncode()
    if flush_bytes and len(flush_bytes) > 0:
        out_file.write(flush_bytes)

    out_file.close()

    elapsed = time.time() - start
    vram = torch.cuda.max_memory_reserved() / 1024**3
    n_frames = max_frames if max_frames > 0 else total_frames

    print(f"\n{'=' * 60}")
    print(f"Encode done: ~{n_frames} frames in {elapsed / 60:.1f} min "
          f"({n_frames / elapsed:.1f} fps)")
    print(f"  Peak VRAM: {vram:.1f}GB")
    if errors[0] > 0:
        print(f"  Errors: {errors[0]}")

    # Per-stage timing
    def avg_ms(total, count):
        return (total / max(count, 1)) * 1000

    print(f"  Timing per frame:")
    print(f"    decode: {avg_ms(t_decode_total[0], n_decoded[0]):.1f}ms "
          f"({n_decoded[0]} frames, {t_decode_total[0]:.2f}s total)")
    print(f"    infer:  {avg_ms(t_infer_total[0], n_inferred[0]):.1f}ms "
          f"({n_inferred[0]} frames, {t_infer_total[0]:.2f}s total)")
    print(f"    encode: {avg_ms(t_encode_total[0], n_encoded[0]):.1f}ms "
          f"({n_encoded[0]} frames, {t_encode_total[0]:.2f}s total)")

    raw_size = os.path.getsize(raw_path) / 1024**2 if os.path.exists(raw_path) else 0
    if raw_size == 0:
        print("WARNING: NVENC produced empty bitstream.")
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return

    print(f"  Raw bitstream: {raw_size:.1f} MB")

    success = wrap_bitstream(raw_path, output_path, input_path, fps_info, codec)

    if os.path.exists(raw_path):
        os.remove(raw_path)

    if success and os.path.exists(output_path):
        final_size = os.path.getsize(output_path) / 1024**2
        print(f"  Final output: {output_path} ({final_size:.1f} MB)")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-copy GPU denoiser v2 (CUDA streams + ring buffers)")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--checkpoint", required=True, help="NAFNet checkpoint path")
    parser.add_argument("--crf", type=int, default=18, help="CRF quality (default: 18)")
    parser.add_argument("--codec", default="hevc", choices=["hevc", "h264"],
                        help="NVENC codec (default: hevc)")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Max frames to process (-1 = all)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (reduce-overhead mode)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    parser.add_argument("--width", type=int, default=64,
                        help="NAFNet channel width (default: 64)")
    parser.add_argument("--middle-blk-num", type=int, default=12,
                        help="Number of middle blocks (default: 12)")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_gpu_v2.mkv"

    denoise_video(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        crf=args.crf,
        codec=args.codec,
        max_frames=args.max_frames,
        fp16=not args.fp32,
        use_compile=args.compile,
        width=args.width,
        middle_blk_num=args.middle_blk_num,
    )


if __name__ == "__main__":
    main()
