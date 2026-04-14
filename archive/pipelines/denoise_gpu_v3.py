"""
GPU pipeline v3 — NVDEC decode + PyTorch inference + ffmpeg pipe encode.

Decode: PyNvVideoCodec NVDEC (GPU, zero-copy via DLPack)
Infer:  NAFNet via torch.compile (GPU, CUDA graph)
Encode: ffmpeg NVENC via stdin pipe (separate process, no GIL contention)

v2 showed that PyNvVideoCodec's Encode() holds the Python GIL during
nvEncLockBitstream (~180ms), serializing all pipeline stages. Moving encode
to ffmpeg (separate process) eliminates GIL contention entirely.

Tradeoff: one GPU→CPU copy per frame (~6MB, ~2-5ms) but eliminates ~140ms
of GIL contention per frame.

Usage:
    python pipelines/denoise_gpu_v3.py --input episode.mkv --checkpoint ckpt.pth --width 32 --middle-blk-num 4 --compile
"""
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from threading import Thread
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True

if sys.platform == "win32":
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

DEVICE = "cuda"


def load_nafnet(checkpoint_path, device="cuda", fp16=True, use_compile=False,
                width=64, middle_blk_num=12, enc_blk_nums=None, dec_blk_nums=None):
    enc_blk_nums = enc_blk_nums or [2, 2, 4, 8]
    dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
    model = NAFNet(
        img_channel=3, width=width, middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
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
            print("  torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
    return model


def denoise_video(
    input_path, output_path, checkpoint_path,
    crf=18, codec="hevc", max_frames=-1, fp16=True,
    use_compile=False, width=64, middle_blk_num=12,
):
    try:
        import PyNvVideoCodec as nvc
    except ImportError as e:
        print(f"ERROR: PyNvVideoCodec not available: {e}")
        sys.exit(1)

    model = load_nafnet(checkpoint_path, device=DEVICE, fp16=fp16,
                        use_compile=use_compile, width=width,
                        middle_blk_num=middle_blk_num)

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    _, _, fps_info, total_frames, duration = get_video_info(input_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # NVDEC on its own stream
    decode_stream = torch.cuda.Stream()
    print(f"\nSetting up NVDEC decoder (stream={decode_stream.cuda_stream})...")
    decoder = nvc.SimpleDecoder(
        input_path, gpu_id=0,
        cuda_stream=decode_stream.cuda_stream,
        output_color_type=nvc.OutputColorType.RGBP,
    )

    frame_iter = iter(decoder)
    try:
        first_raw = next(frame_iter)
    except StopIteration:
        print("ERROR: Could not decode any frames.")
        sys.exit(1)

    first_frame = torch.from_dlpack(first_raw)
    _, h, w = first_frame.shape
    print(f"  Decoded first frame: {w}x{h}")

    h_pad = h + (16 - h % 16) % 16
    w_pad = w + (16 - w % 16) % 16
    dtype = torch.float16 if fp16 else torch.float32

    # Warmup on default stream (where CUDA graph will replay)
    if use_compile:
        print(f"  Warming up at {w_pad}x{h_pad}...")
        dummy = torch.randn(1, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)
        dummy = dummy.to(memory_format=torch.channels_last)
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
            _ = model(dummy)
        torch.cuda.synchronize()
        print(f"  Compile warmup: {time.time() - t0:.1f}s")
        del dummy
        torch.cuda.empty_cache()

    # --- ffmpeg NVENC (separate process, no GIL) ---
    ffmpeg = get_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video_only = output_path + ".video.mkv"
    nvenc_codec = "hevc_nvenc" if codec == "hevc" else "h264_nvenc"
    enc_cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}", "-framerate", str(fps_info),
        "-i", "pipe:0",
        "-c:v", nvenc_codec, "-rc", "vbr", "-cq", str(crf),
        "-preset", "p4", "-tune", "hq", "-pix_fmt", "yuv420p",
        video_only,
    ]
    encoder_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=w * h * 3 * 4)
    print(f"  ffmpeg NVENC: {nvenc_codec} p4 cq={crf}, pid={encoder_proc.pid}")

    # --- Decode thread ---
    SENTINEL = object()
    decode_q = Queue(maxsize=8)

    def decode_thread_fn():
        count = 0
        with torch.cuda.stream(decode_stream):
            decode_q.put(first_frame.clone())
        count += 1
        for raw_frame in frame_iter:
            if 0 < max_frames <= count:
                break
            with torch.cuda.stream(decode_stream):
                decode_q.put(torch.from_dlpack(raw_frame).clone())
            count += 1
        decode_q.put(SENTINEL)

    # Pre-allocate buffers
    input_buf = torch.zeros(1, 3, h_pad, w_pad, dtype=dtype, device=DEVICE)
    input_buf = input_buf.to(memory_format=torch.channels_last)

    print(f"\nProcessing: {'fp16' if fp16 else 'fp32'}, "
          f"{'compiled' if use_compile else 'eager'}, {nvenc_codec}")
    print(f"Input: {input_path}")
    print(f"  {w}x{h} @ {fps_info:.3f}fps, ~{total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    start = time.time()
    t_infer = 0.0
    t_xfer = 0.0
    t_write = 0.0

    t_dec = Thread(target=decode_thread_fn, name="decode", daemon=True)
    t_dec.start()

    processed = 0
    with torch.no_grad():
        while True:
            item = decode_q.get()
            if item is SENTINEL:
                break

            t0 = time.perf_counter()
            input_buf[0, :, :h, :w] = item.to(dtype=dtype) / 255.0
            del item
            out = model(input_buf)
            out_rgb = (out[0, :, :h, :w].clamp(0, 1) * 255).byte()
            t1 = time.perf_counter()
            t_infer += t1 - t0

            # GPU → CPU (includes sync)
            out_cpu = out_rgb.permute(1, 2, 0).contiguous().cpu()
            t2 = time.perf_counter()
            t_xfer += t2 - t1

            # Pipe to ffmpeg
            encoder_proc.stdin.write(out_cpu.numpy().tobytes())
            t3 = time.perf_counter()
            t_write += t3 - t2

            processed += 1
            if processed % 50 == 0 or processed == 1:
                elapsed = time.time() - start
                fps_actual = processed / elapsed
                vram = torch.cuda.max_memory_reserved() / 1024**3
                eta_min = (total_frames - processed) / max(fps_actual, 0.01) / 60
                print(f"  [{processed}/{total_frames}] "
                      f"{fps_actual:.1f} fps, VRAM: {vram:.1f}GB, ETA: {eta_min:.1f}min")

    t_dec.join()
    encoder_proc.stdin.close()
    encoder_proc.wait()

    elapsed = time.time() - start
    vram = torch.cuda.max_memory_reserved() / 1024**3

    print(f"\n{'=' * 60}")
    print(f"Done: {processed} frames in {elapsed / 60:.1f} min "
          f"({processed / elapsed:.1f} fps)")
    print(f"  Peak VRAM: {vram:.1f}GB")
    avg = lambda t: (t / max(processed, 1)) * 1000
    print(f"  Per frame: infer={avg(t_infer):.1f}ms "
          f"xfer={avg(t_xfer):.1f}ms write={avg(t_write):.1f}ms")

    # Mux audio
    if os.path.exists(video_only) and os.path.getsize(video_only) > 0:
        mux_cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_only, "-i", input_path,
            "-map", "0:v", "-map", "1:a?", "-c", "copy", output_path,
        ]
        print(f"\nMuxing audio...")
        r = subprocess.run(mux_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            os.replace(video_only, output_path)
        else:
            os.remove(video_only)
        final_size = os.path.getsize(output_path) / 1024**2
        print(f"  Final: {output_path} ({final_size:.1f} MB)")
    else:
        print("WARNING: ffmpeg produced no output.")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="GPU decode + infer, ffmpeg NVENC encode")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--codec", default="hevc", choices=["hevc", "h264"])
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--middle-blk-num", type=int, default=12)
    args = parser.parse_args()
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = f"{base}_v3.mkv"
    denoise_video(
        input_path=args.input, output_path=args.output,
        checkpoint_path=args.checkpoint, crf=args.crf, codec=args.codec,
        max_frames=args.max_frames, fp16=not args.fp32,
        use_compile=args.compile, width=args.width,
        middle_blk_num=args.middle_blk_num,
    )

if __name__ == "__main__":
    main()
