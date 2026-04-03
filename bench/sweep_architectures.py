"""
Sweep NAFNet architecture variants to find the fastest config that might
still produce acceptable quality at 1080p denoising.

Target: 30+ fps (33ms/frame) on RTX 3060 Laptop GPU.

Usage:
  python bench/sweep_architectures.py
  python bench/sweep_architectures.py --target-fps 24    # lower target
  python bench/sweep_architectures.py --no-compile        # skip torch.compile
"""
import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, ".")
from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile


CONFIGS = [
    # name, width, middle_blk_num, enc_blk_nums, dec_blk_nums
    ("w32_mid4 (current)",  32, 4, [2,2,4,8], [2,2,2,2]),
    ("w32_mid2",            32, 2, [1,1,2,4], [1,1,1,1]),
    ("w32_mid1",            32, 1, [1,1,1,2], [1,1,1,1]),
    ("w32_min",             32, 1, [1,1,1,1], [1,1,1,1]),
    ("w24_mid4",            24, 4, [2,2,4,8], [2,2,2,2]),
    ("w24_mid2",            24, 2, [1,1,2,4], [1,1,1,1]),
    ("w24_mid1",            24, 1, [1,1,1,2], [1,1,1,1]),
    ("w24_min",             24, 1, [1,1,1,1], [1,1,1,1]),
    ("w16_mid4",            16, 4, [2,2,4,8], [2,2,2,2]),
    ("w16_mid2",            16, 2, [1,1,2,4], [1,1,1,1]),
    ("w16_mid1",            16, 1, [1,1,1,2], [1,1,1,1]),
    ("w16_min",             16, 1, [1,1,1,1], [1,1,1,1]),
    ("w8_mid2",              8, 2, [1,1,2,4], [1,1,1,1]),
    ("w8_min",               8, 1, [1,1,1,1], [1,1,1,1]),
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_config(name, width, middle_blk_num, enc_blk_nums, dec_blk_nums,
                     use_compile=True, n_warmup=3, n_bench=20):
    """Benchmark a single architecture config. Returns (fps, ms_per_frame, params, vram_mb)."""
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = NAFNet(
            img_channel=3, width=width, middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
        )
        params = count_params(model)
        model.eval()
        model = swap_layernorm_for_compile(model)
        model.half()
        model = model.to(memory_format=torch.channels_last).cuda()

        if use_compile:
            torch._inductor.config.conv_1x1_as_mm = True
            torch.backends.cudnn.benchmark = True
            model = torch.compile(model, mode="reduce-overhead")

        dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)
        dummy = dummy.to(memory_format=torch.channels_last)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(dummy)
        torch.cuda.synchronize()

        vram_mb = torch.cuda.max_memory_reserved() / 1024 / 1024

        # Benchmark with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(n_bench):
                _ = model(dummy)
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        ms_per_frame = total_ms / n_bench
        fps = 1000.0 / ms_per_frame

        return fps, ms_per_frame, params, vram_mb

    except torch.cuda.OutOfMemoryError:
        return 0, float("inf"), 0, 0
    finally:
        # Clean up for next config
        del model
        torch.compiler.reset()
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Sweep NAFNet architecture variants")
    parser.add_argument("--target-fps", type=float, default=30.0,
                        help="Target fps (default: 30)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (faster sweep, less accurate)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Target: {args.target_fps} fps ({1000/args.target_fps:.1f} ms/frame)")
    print(f"Compile: {'yes' if not args.no_compile else 'no (eager)'}")
    print()

    print(f"{'Config':<22} {'Params':>8} {'ms/frame':>10} {'FPS':>8} {'VRAM':>8} {'Status'}")
    print("-" * 75)

    results = []
    for name, width, mid, enc, dec in CONFIGS:
        fps, ms, params, vram = benchmark_config(
            name, width, mid, enc, dec,
            use_compile=not args.no_compile,
        )

        if fps > 0:
            status = "TARGET" if fps >= args.target_fps else ""
            params_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
            print(f"  {name:<20} {params_str:>8} {ms:>8.1f}ms {fps:>7.1f} {vram:>6.0f}MB  {status}")
            results.append((name, width, mid, enc, dec, fps, ms, params, vram))
        else:
            print(f"  {name:<20}      OOM")

    print()
    print("=" * 75)

    # Find fastest config that we'd want to train
    viable = [(n, f, m, p) for n, w, mid, enc, dec, f, m, p, v in results if f >= args.target_fps]
    if viable:
        print(f"\nConfigs meeting {args.target_fps} fps target:")
        for name, fps, ms, params in sorted(viable, key=lambda x: -x[3]):
            params_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
            print(f"  {name}: {fps:.1f} fps, {params_str} params (largest = best quality candidate)")
    else:
        print(f"\nNo config meets {args.target_fps} fps. Fastest was:")
        fastest = max(results, key=lambda x: x[5])
        print(f"  {fastest[0]}: {fastest[5]:.1f} fps ({fastest[6]:.1f}ms), {fastest[7]/1e6:.1f}M params")
        print(f"\nConsider: --target-fps {fastest[5]:.0f}, or reduce resolution, or try alternative architectures")


if __name__ == "__main__":
    main()
