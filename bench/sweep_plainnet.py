"""
Sweep PlainDenoise and UNetDenoise architecture variants at 1080p.

These are INT8-native architectures (Conv+BN+ReLU only). FP16 results here
give a lower bound — INT8 should be 2-4x faster on Ampere tensor cores.

Usage:
  python bench/sweep_plainnet.py
  python bench/sweep_plainnet.py --no-compile     # eager mode (faster sweep)
  python bench/sweep_plainnet.py --target-fps 24  # lower target
"""
import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, ".")
from lib.plainnet_arch import PlainDenoise, UNetDenoise, count_params


# (name, constructor, kwargs)
CONFIGS = [
    # --- PlainDenoise: sweep nc (channels) × nb (layers) ---
    # Large: high quality candidates
    ("Plain nc=96 nb=15",  PlainDenoise, dict(nc=96, nb=15)),
    ("Plain nc=80 nb=15",  PlainDenoise, dict(nc=80, nb=15)),
    ("Plain nc=64 nb=20",  PlainDenoise, dict(nc=64, nb=20)),
    ("Plain nc=64 nb=15",  PlainDenoise, dict(nc=64, nb=15)),

    # Medium: balanced speed/quality
    ("Plain nc=64 nb=12",  PlainDenoise, dict(nc=64, nb=12)),
    ("Plain nc=48 nb=15",  PlainDenoise, dict(nc=48, nb=15)),
    ("Plain nc=48 nb=12",  PlainDenoise, dict(nc=48, nb=12)),
    ("Plain nc=48 nb=10",  PlainDenoise, dict(nc=48, nb=10)),

    # Small: speed-first
    ("Plain nc=32 nb=15",  PlainDenoise, dict(nc=32, nb=15)),
    ("Plain nc=32 nb=12",  PlainDenoise, dict(nc=32, nb=12)),
    ("Plain nc=32 nb=10",  PlainDenoise, dict(nc=32, nb=10)),
    ("Plain nc=32 nb=8",   PlainDenoise, dict(nc=32, nb=8)),

    # Tiny: INT8 speed demons
    ("Plain nc=24 nb=10",  PlainDenoise, dict(nc=24, nb=10)),
    ("Plain nc=16 nb=10",  PlainDenoise, dict(nc=16, nb=10)),

    # --- UNetDenoise: better receptive field ---
    ("UNet nc=48 mid=2",   UNetDenoise, dict(nc=48, nb_enc=(2,2), nb_dec=(2,2), nb_mid=2)),
    ("UNet nc=48 mid=4",   UNetDenoise, dict(nc=48, nb_enc=(2,3), nb_dec=(2,2), nb_mid=4)),
    ("UNet nc=32 mid=2",   UNetDenoise, dict(nc=32, nb_enc=(2,2), nb_dec=(2,2), nb_mid=2)),
    ("UNet nc=32 mid=4",   UNetDenoise, dict(nc=32, nb_enc=(2,3), nb_dec=(2,2), nb_mid=4)),
    ("UNet nc=64 mid=2",   UNetDenoise, dict(nc=64, nb_enc=(2,2), nb_dec=(2,2), nb_mid=2)),
]


def benchmark_config(name, cls, kwargs, use_compile=True, n_warmup=3, n_bench=20):
    """Benchmark a single config. Returns (fps, ms, params, vram_mb)."""
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = cls(**kwargs)
        params = count_params(model)

        # Fuse reparam branches (simulates inference deployment)
        model.fuse_reparam()
        model.eval()
        model.half()
        model = model.to(memory_format=torch.channels_last).cuda()

        if use_compile:
            torch._inductor.config.conv_1x1_as_mm = True
            torch.backends.cudnn.benchmark = True
            model = torch.compile(model, mode="reduce-overhead")

        dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)
        dummy = dummy.to(memory_format=torch.channels_last)

        # Warmup (extra for torch.compile)
        for _ in range(n_warmup + (3 if use_compile else 0)):
            with torch.no_grad():
                _ = model(dummy)
        torch.cuda.synchronize()

        vram_mb = torch.cuda.max_memory_reserved() / 1024 / 1024

        # Benchmark with CUDA events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(n_bench):
                _ = model(dummy)
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        ms = total_ms / n_bench
        fps = 1000.0 / ms

        return fps, ms, params, vram_mb

    except torch.cuda.OutOfMemoryError:
        return 0, float("inf"), 0, 0
    finally:
        try:
            del model
        except NameError:
            pass
        torch.compiler.reset()
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Sweep PlainDenoise/UNetDenoise configs")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Target: {args.target_fps} fps ({1000/args.target_fps:.1f} ms/frame)")
    print(f"Compile: {'yes' if not args.no_compile else 'no (eager)'}")
    print(f"NOTE: These are FP16 results. INT8 should be 2-4x faster on Ampere.\n")

    # Reference: NAFNet w8_mid2 = 37 fps, w32_mid4 = 5.3 fps (from sweep_architectures.py)
    print(f"{'Config':<24} {'Params':>8} {'ms/frame':>10} {'FPS':>8} {'VRAM':>8} {'INT8 est':>10} {'Status'}")
    print("-" * 90)

    results = []
    for name, cls, kwargs in CONFIGS:
        fps, ms, params, vram = benchmark_config(
            name, cls, kwargs, use_compile=not args.no_compile
        )

        if fps > 0:
            # Conservative INT8 estimate: 2x FP16 speed (Ampere tensor cores)
            int8_fps = fps * 2.0
            status = "TARGET" if fps >= args.target_fps else (
                "INT8-TARGET" if int8_fps >= args.target_fps else "")
            p_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
            print(f"  {name:<22} {p_str:>8} {ms:>8.1f}ms {fps:>7.1f} {vram:>6.0f}MB"
                  f"  ~{int8_fps:.0f} fps  {status}")
            results.append((name, fps, ms, params, vram, int8_fps))
        else:
            print(f"  {name:<22}      OOM")

    print()
    print("=" * 90)

    # Analysis
    fp16_viable = [(n, f, m, p) for n, f, m, p, v, i in results if f >= args.target_fps]
    int8_viable = [(n, f, m, p, i) for n, f, m, p, v, i in results
                   if i >= args.target_fps and f < args.target_fps]

    if fp16_viable:
        print(f"\nConfigs meeting {args.target_fps} fps in FP16 (no quantization needed):")
        for name, fps, ms, params in sorted(fp16_viable, key=lambda x: -x[3]):
            p_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
            print(f"  {name}: {fps:.1f} fps, {p_str} params")

    if int8_viable:
        print(f"\nConfigs estimated to meet {args.target_fps} fps with INT8:")
        for name, fps, ms, params, int8 in sorted(int8_viable, key=lambda x: -x[3]):
            p_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
            print(f"  {name}: {fps:.1f} fps FP16, ~{int8:.0f} fps INT8, {p_str} params")

    if not fp16_viable and not int8_viable:
        print(f"\nNo config meets target. Consider lower resolution or more aggressive pruning.")

    # Best quality candidate for INT8
    best = max(results, key=lambda x: x[3] if x[5] >= args.target_fps else 0)
    if best[5] >= args.target_fps:
        p_str = f"{best[3]/1e6:.1f}M" if best[3] > 1e6 else f"{best[3]/1e3:.0f}K"
        print(f"\nRecommended: {best[0]} ({p_str} params)")
        print(f"  FP16: {best[1]:.1f} fps, INT8 estimate: ~{best[5]:.0f} fps")
        print(f"  Largest model within INT8 target = best quality potential")


if __name__ == "__main__":
    main()
