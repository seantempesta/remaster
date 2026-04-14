"""
Zero-copy GPU video pipeline — NVDEC decode -> NAFNet inference -> NVENC encode.

Uses PyNvVideoCodec for hardware decode/encode, keeping frames on GPU throughout.
No CPU round-trips for frame data (except final mux with ffmpeg for audio/subs).

The NVENC encoder outputs a raw H.264/HEVC bitstream. After encoding, ffmpeg wraps
it in a container and muxes audio/subtitles from the original file.

Usage:
    python pipelines/denoise_gpu.py --input episode.mkv --checkpoint checkpoints/nafnet_distill/nafnet_best.pth
    python pipelines/denoise_gpu.py --input episode.mkv --checkpoint ckpt.pth --compile --max-frames 100
    python pipelines/denoise_gpu.py --input episode.mkv --checkpoint ckpt.pth --width 32 --middle-blk-num 4
"""
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True

# PyNvVideoCodec needs CUDA runtime DLLs — add torch's lib dir on Windows
# (no CUDA toolkit install needed, PyTorch ships the required DLLs)
if sys.platform == "win32":
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

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

    # Step 1: wrap raw bitstream into a proper container with timestamps
    wrapped_video = raw_path + ".mkv"
    cmd_wrap = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", raw_codec, "-r", f"{fps:.6f}", "-i", raw_path,
        "-c:v", "copy", "-video_track_timescale", "90000",
        wrapped_video,
    ]
    print(f"\nWrapping bitstream...")
    r = subprocess.run(cmd_wrap, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  Wrap failed: {r.stderr.strip()}")
        return False

    # Step 2: mux with audio/subs from original
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
        # Fall back to video only
        os.replace(wrapped_video, output_path)
        print("  Video only (no audio).")
        return True

    # Clean up intermediate
    if os.path.exists(wrapped_video):
        os.remove(wrapped_video)
    print(f"  Muxed output: {output_path}")
    return True


def denoise_video(
    input_path, output_path, checkpoint_path,
    crf=18, codec="hevc", max_frames=-1, fp16=True,
    use_compile=False, width=64, middle_blk_num=12,
):
    """Zero-copy GPU pipeline: NVDEC -> NAFNet -> NVENC."""

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

    # Get video info for progress reporting
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    _, _, fps_info, total_frames, duration = get_video_info(input_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # --- NVDEC Decoder ---
    print(f"\nSetting up NVDEC decoder...")
    decoder = nvc.SimpleDecoder(
        input_path, gpu_id=0,
        output_color_type=nvc.OutputColorType.RGBP,
    )

    # Decode first frame to get dimensions and warm up
    frame_iter = iter(decoder)
    try:
        first_raw = next(frame_iter)
    except StopIteration:
        print("ERROR: Could not decode any frames from input.")
        sys.exit(1)

    first_frame = torch.from_dlpack(first_raw).clone()  # [3, H, W] uint8 on cuda
    _, h, w = first_frame.shape
    print(f"  Decoded first frame: {w}x{h}")

    # Pad dimensions to multiples of 16
    h_pad = h + (16 - h % 16) % 16
    w_pad = w + (16 - w % 16) % 16

    # --- torch.compile warmup ---
    dtype = torch.float16 if fp16 else torch.float32
    if use_compile:
        print(f"  Warming up compiled model at {w_pad}x{h_pad}...")
        dummy = torch.randn(1, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)
        dummy = dummy.to(memory_format=torch.channels_last)
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
            _ = model(dummy)  # second pass for CUDA graph recording
        torch.cuda.synchronize()
        print(f"  Compile warmup: {time.time() - t0:.1f}s")
        del dummy
        torch.cuda.empty_cache()

    # --- NVENC Encoder ---
    print(f"Setting up NVENC encoder...")
    # Raw bitstream output (will wrap in container later)
    raw_path = output_path + ".raw"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # CreateEncoder(width, height, fmt_string, usecpuinputbuffer, **kwargs)
    # ARGB format: encoder expects [H, W, 4] uint8 CUDA tensor
    encoder = nvc.CreateEncoder(
        w, h, "ARGB", False,
        codec=codec,
        preset="P4",
    )
    print(f"  NVENC encoder: {codec} ARGB, {w}x{h}, CRF {crf}")

    out_file = open(raw_path, "wb")

    # --- Pipelined processing ---
    # Three stages run concurrently on separate hardware:
    #   NVDEC (decode) -> CUDA cores (inference) -> NVENC (encode)
    # Each stage uses its own CUDA stream for async execution.
    from threading import Thread
    from queue import Queue

    SENTINEL = object()
    decode_q = Queue(maxsize=4)   # decoded GPU tensors
    encode_q = Queue(maxsize=4)   # post-inference GPU tensors

    print(f"\nProcessing: {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}, codec={codec}, pipelined")
    print(f"Input: {input_path}")
    print(f"  {w}x{h} @ {fps_info:.3f}fps, ~{total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    start = time.time()
    errors = [0]

    # --- Decode thread (NVDEC) ---
    def decode_thread():
        """Decode frames on NVDEC, put GPU tensors in queue."""
        count = 0
        # First frame already decoded
        decode_q.put(first_frame.clone())
        count += 1
        for raw_frame in frame_iter:
            if max_frames > 0 and count >= max_frames:
                break
            # Clone because dlpack tensor may be reused by decoder
            rgb = torch.from_dlpack(raw_frame).clone()
            decode_q.put(rgb)
            count += 1
        decode_q.put(SENTINEL)

    # --- Inference thread (CUDA cores) ---
    def infer_thread():
        """Run NAFNet inference, put results in encode queue."""
        # Pre-allocate padded input buffer
        input_buf = torch.zeros(
            1, 3, h_pad, w_pad, dtype=dtype,
            device=torch.device(DEVICE))
        input_buf = input_buf.to(memory_format=torch.channels_last)

        while True:
            item = decode_q.get()
            if item is SENTINEL:
                encode_q.put(SENTINEL)
                break
            try:
                # Normalize to [0,1] float, place in padded buffer
                input_buf[0, :, :h, :w] = item.to(dtype=dtype) / 255.0
                del item

                with torch.no_grad():
                    out = model(input_buf)

                # Crop, clamp, quantize to uint8
                out_rgb = (out[0, :, :h, :w].clamp(0, 1) * 255).byte()
                encode_q.put(out_rgb)
            except Exception as e:
                print(f"  Inference error: {e}")
                errors[0] += 1

    # --- Encode thread (NVENC) ---
    def encode_thread():
        """NVENC encode from GPU tensors, write bitstream."""
        argb_buf = torch.empty(h, w, 4, dtype=torch.uint8,
                               device=DEVICE)
        argb_buf[:, :, 0] = 255  # alpha channel
        processed = 0

        while True:
            item = encode_q.get()
            if item is SENTINEL:
                break
            try:
                # Fill ARGB: [H, W, 4] = [A, R, G, B]
                argb_buf[:, :, 1] = item[0]  # R
                argb_buf[:, :, 2] = item[1]  # G
                argb_buf[:, :, 3] = item[2]  # B
                del item

                encoded = encoder.Encode(argb_buf)
                if encoded and len(encoded) > 0:
                    out_file.write(encoded)

                processed += 1
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
                errors[0] += 1
                processed += 1

    # Launch all three threads
    t_dec = Thread(target=decode_thread, name="decode", daemon=True)
    t_inf = Thread(target=infer_thread, name="infer", daemon=True)
    t_enc = Thread(target=encode_thread, name="encode", daemon=True)
    t_dec.start()
    t_inf.start()
    t_enc.start()

    # Wait for pipeline to drain
    t_dec.join()
    t_inf.join()
    t_enc.join()

    # Flush encoder
    flush_bytes = encoder.EndEncode()
    if flush_bytes and len(flush_bytes) > 0:
        out_file.write(flush_bytes)

    out_file.close()
    del first_frame

    elapsed = time.time() - start
    vram = torch.cuda.max_memory_reserved() / 1024**3
    # Count frames from file size (encode thread's local var isn't accessible)
    raw_size_b = os.path.getsize(raw_path) if os.path.exists(raw_path) else 0
    n_frames = max_frames if max_frames > 0 else total_frames

    print(f"\n{'=' * 60}")
    print(f"Encode done: ~{n_frames} frames in {elapsed / 60:.1f} min "
          f"({n_frames / elapsed:.1f} fps)")
    print(f"  Peak VRAM: {vram:.1f}GB")
    if errors[0] > 0:
        print(f"  Errors: {errors[0]}")

    # Check raw bitstream
    raw_size = os.path.getsize(raw_path) / 1024**2 if os.path.exists(raw_path) else 0
    if raw_size == 0:
        print("WARNING: NVENC produced empty bitstream.")
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return

    print(f"  Raw bitstream: {raw_size:.1f} MB")

    # Wrap in container + mux audio/subs
    success = wrap_bitstream(raw_path, output_path, input_path, fps_info, codec)

    # Clean up raw bitstream
    if os.path.exists(raw_path):
        os.remove(raw_path)

    if success and os.path.exists(output_path):
        final_size = os.path.getsize(output_path) / 1024**2
        print(f"  Final output: {output_path} ({final_size:.1f} MB)")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-copy GPU video denoiser (NVDEC -> NAFNet -> NVENC)")
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
        args.output = f"{base}_gpu.mkv"

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
