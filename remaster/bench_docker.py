"""
Remaster Docker benchmark — verify GPU performance inside the container.

Tests:
1. torch.compile (Inductor) baseline — should match native Windows (~78 fps)
2. AOT Inductor compilation — compile to standalone .so
3. Full pipeline: decode → inference → encode

Usage (inside Docker container):
  python remaster/bench_docker.py
  python remaster/bench_docker.py --aot      # also test AOT compilation
  python remaster/bench_docker.py --pipeline  # also test full pipeline
"""
import argparse
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench_torch_compile():
    """Benchmark torch.compile with Inductor backend."""
    import torch

    print("=== torch.compile (Inductor) Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print()

    from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile

    model = NAFNet(img_channel=3, width=32, middle_blk_num=4,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

    ckpt_path = "checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model.load_state_dict(state, strict=False)
        del ckpt, state
        gc.collect()
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"WARNING: No weights at {ckpt_path}, using random weights")

    model.eval()
    model = swap_layernorm_for_compile(model)
    model.half().cuda()

    dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)

    print("Compiling (first run triggers Triton kernel generation)...")
    compiled = torch.compile(model, mode="reduce-overhead")

    # Warmup (triggers compilation + CUDA graph capture)
    for i in range(5):
        with torch.no_grad():
            _ = compiled(dummy)
    torch.cuda.synchronize()
    print("Warmup done.")

    # Benchmark
    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = compiled(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = N / elapsed
    latency_ms = elapsed / N * 1000
    print(f"\n  torch.compile: {fps:.1f} fps ({latency_ms:.1f} ms/frame)")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")

    target = 72
    status = "TARGET MET" if fps >= target else "BELOW TARGET"
    print(f"  Status: [{status}] (target: {target}+ fps)")
    print()

    return fps


def bench_aot_compile():
    """Benchmark AOT Inductor compilation."""
    import torch

    print("=== AOT Inductor Compilation ===")

    from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile

    model = NAFNet(img_channel=3, width=32, middle_blk_num=4,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

    ckpt_path = "checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model.load_state_dict(state, strict=False)
        del ckpt, state
        gc.collect()

    model.eval()
    model = swap_layernorm_for_compile(model)
    model.half().cuda()

    dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)

    # Export
    print("Exporting model graph...")
    exported = torch.export.export(model, (dummy,), strict=False)

    # AOT compile
    print("AOT compiling (generating .so with fused CUDA kernels)...")
    t0 = time.perf_counter()
    so_path = torch._inductor.aot_compile(exported.module(), (dummy,))
    compile_time = time.perf_counter() - t0
    so_size = os.path.getsize(so_path) / 1024 / 1024
    print(f"  Compiled in {compile_time:.1f}s")
    print(f"  Output: {so_path} ({so_size:.1f} MB)")

    # Load and benchmark
    print("Loading AOT compiled model...")
    runner = torch._export.aot_load(so_path, device="cuda")

    for _ in range(5):
        with torch.no_grad():
            _ = runner(dummy)
    torch.cuda.synchronize()

    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = runner(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = N / elapsed
    print(f"\n  AOT Inductor: {fps:.1f} fps ({elapsed / N * 1000:.1f} ms/frame)")
    print()

    # Copy artifact to persistent location
    output_dir = "checkpoints/nafnet_w32_mid4/aot_local/"
    os.makedirs(output_dir, exist_ok=True)
    import shutil
    so_dir = os.path.dirname(so_path)
    for f in os.listdir(so_dir):
        src = os.path.join(so_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, f))
    print(f"  Artifacts saved to {output_dir}")
    print()

    return fps


def bench_pipeline():
    """Benchmark full decode -> inference -> encode pipeline."""
    import torch
    import subprocess

    print("=== Full Pipeline Benchmark ===")

    test_clip = "/videos/input/clip_mid_1080p.mp4"
    if not os.path.exists(test_clip):
        # Try local data dir
        test_clip = "data/clip_mid_1080p.mp4"
    if not os.path.exists(test_clip):
        print("  SKIP: No test video found. Mount a video directory.")
        return 0

    from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile

    model = NAFNet(img_channel=3, width=32, middle_blk_num=4,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

    ckpt_path = "checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model.load_state_dict(state, strict=False)
        del ckpt, state
        gc.collect()

    model.eval()
    model = swap_layernorm_for_compile(model)
    model.half().cuda()
    compiled = torch.compile(model, mode="reduce-overhead")

    # Warmup
    dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)
    for _ in range(3):
        with torch.no_grad():
            _ = compiled(dummy)
    torch.cuda.synchronize()

    # Decode with ffmpeg
    print(f"  Input: {test_clip}")

    # Get video info
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", test_clip],
        capture_output=True, text=True,
    )

    import json
    info = json.loads(probe.stdout)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(vstream["width"])
    height = int(vstream["height"])
    nb_frames = int(vstream.get("nb_frames", 720))
    print(f"  Resolution: {width}x{height}, Frames: {nb_frames}")

    # Padded dimensions
    target_h = ((height + 15) // 16) * 16
    target_w = ((width + 15) // 16) * 16
    pad_h = target_h - height
    pad_w = target_w - width

    # Decode → process → encode pipeline using subprocess pipes
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", test_clip,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-"
    ]

    output_path = "/tmp/bench_pipeline_output.mkv"
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "24",
        "-i", "-",
        "-c:v", "libx265", "-crf", "20", "-preset", "ultrafast",
        output_path,
    ]

    frame_size = width * height * 3  # RGB24

    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frames_processed = 0
    t0 = time.perf_counter()

    while True:
        raw = decode_proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        # Convert to tensor
        import numpy as np
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.half().cuda() / 255.0

        # Pad
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

        # Inference
        with torch.no_grad():
            out = compiled(tensor)

        # Crop and convert back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :height, :width]

        out_np = (out.squeeze(0).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        encode_proc.stdin.write(out_np.tobytes())
        frames_processed += 1

    decode_proc.wait()
    encode_proc.stdin.close()
    encode_proc.wait()
    elapsed = time.perf_counter() - t0

    fps = frames_processed / elapsed
    print(f"\n  Processed {frames_processed} frames in {elapsed:.1f}s")
    print(f"  Pipeline: {fps:.1f} fps ({elapsed / max(frames_processed, 1) * 1000:.1f} ms/frame)")

    if os.path.exists(output_path):
        out_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Output: {out_size:.1f} MB")
        os.remove(output_path)

    print()
    return fps


def main():
    parser = argparse.ArgumentParser(description="Remaster Docker GPU benchmark")
    parser.add_argument("--aot", action="store_true", help="Also test AOT Inductor compilation")
    parser.add_argument("--pipeline", action="store_true", help="Also test full pipeline")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all:
        args.aot = True
        args.pipeline = True

    compile_fps = bench_torch_compile()

    if args.aot:
        aot_fps = bench_aot_compile()

    if args.pipeline:
        pipeline_fps = bench_pipeline()

    print("=== Summary ===")
    print(f"  torch.compile: {compile_fps:.1f} fps")
    if args.aot:
        print(f"  AOT Inductor:  {aot_fps:.1f} fps")
    if args.pipeline:
        print(f"  Full pipeline: {pipeline_fps:.1f} fps")


if __name__ == "__main__":
    main()
