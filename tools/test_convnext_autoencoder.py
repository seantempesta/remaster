"""
Test ConvNeXt-V2 Autoencoder: weight loading, forward pass, masking, param count.

Usage:
  python tools/test_convnext_autoencoder.py
  python tools/test_convnext_autoencoder.py --variant femto
  python tools/test_convnext_autoencoder.py --variant atto --no-pretrained --skip-gpu
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def fmt_params(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.1f}K"


def test_construction(variant, pretrained):
    """Test model construction and pretrained weight loading."""
    from lib.convnext_autoencoder import ConvNeXtV2Autoencoder

    print(f"\n{'='*60}")
    print(f"  ConvNeXt-V2 Autoencoder - {variant.upper()}")
    print(f"{'='*60}")

    model = ConvNeXtV2Autoencoder.from_config(variant, pretrained=pretrained)

    print(f"\n  Total params:   {fmt_params(model.param_count)}")
    print(f"  Encoder params: {fmt_params(model.encoder_param_count)}")
    print(f"  Decoder params: {fmt_params(model.decoder_param_count)}")
    tail_params = model.param_count - model.encoder_param_count - model.decoder_param_count
    print(f"  Tail params:    {fmt_params(tail_params)}")
    print(f"  Input channels: {model.total_in} (3 current + 3 prev + 1 mask)")
    print(f"  Patch size:     {model.patch_size}")

    return model


def test_forward_cpu(model, height=256, width=256):
    """Test forward pass on CPU with small resolution."""
    import torch

    print(f"\n--- CPU forward pass ({height}x{width}) ---")
    model.eval()

    B = 1
    current = torch.randn(B, 3, height, width)
    prev_clean = torch.randn(B, 3, height, width)

    # Test 1: No masking (inference mode)
    with torch.no_grad():
        t0 = time.perf_counter()
        output, mask = model(current, prev_clean, mask_ratio=0.0)
        dt = time.perf_counter() - t0
    print(f"  No mask:    output {list(output.shape)}, mask sum={mask.sum().item():.0f}, {dt*1000:.0f}ms")
    assert output.shape == current.shape, f"Shape mismatch: {output.shape} vs {current.shape}"

    # Test 2: 60% masking (training mode)
    with torch.no_grad():
        t0 = time.perf_counter()
        output, mask = model(current, prev_clean, mask_ratio=0.6)
        dt = time.perf_counter() - t0
    total_patches = (height // model.patch_size) * (width // model.patch_size)
    masked_patches = mask[0, 0, ::model.patch_size, ::model.patch_size].sum().item()
    print(f"  60% mask:   output {list(output.shape)}, "
          f"masked {masked_patches:.0f}/{total_patches} patches, {dt*1000:.0f}ms")
    assert output.shape == current.shape

    # Test 3: Cold start (no previous frame)
    with torch.no_grad():
        t0 = time.perf_counter()
        output, mask = model(current, prev_clean=None, mask_ratio=0.3)
        dt = time.perf_counter() - t0
    print(f"  Cold start: output {list(output.shape)}, {dt*1000:.0f}ms")
    assert output.shape == current.shape

    print("  All CPU tests passed")


def test_forward_gpu(model, height=1080, width=1920):
    """Test forward pass on GPU at full resolution with FP16 autocast."""
    import torch

    if not torch.cuda.is_available():
        print("\n--- GPU test skipped (no CUDA) ---")
        return

    print(f"\n--- GPU forward pass ({height}x{width}, FP16 autocast) ---")
    device = torch.device("cuda")
    model = model.float().to(device)  # Keep model FP32, autocast handles FP16
    model.eval()

    B = 1
    current = torch.randn(B, 3, height, width, device=device)
    prev_clean = torch.randn(B, 3, height, width, device=device)

    # Warmup
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _ = model(current, prev_clean, mask_ratio=0.0)
    torch.cuda.synchronize()

    # Benchmark inference (no masking)
    times = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output, mask = model(current, prev_clean, mask_ratio=0.0)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_ms = sum(times) * 1000 / len(times)
    fps = 1000 / avg_ms
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Inference:  {avg_ms:.1f}ms avg ({fps:.1f} fps)")
    print(f"  VRAM peak:  {vram_mb:.0f}MB")
    print(f"  Output:     {list(output.shape)}, dtype={output.dtype}")
    assert output.shape == (B, 3, height, width)

    # Benchmark with masking (training mode)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output, mask = model(current, prev_clean, mask_ratio=0.6)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
    print(f"  60% masked: {dt:.1f}ms")

    print("  All GPU tests passed")

    # Clean up
    model.cpu()
    del current, prev_clean, output, mask
    torch.cuda.empty_cache()
    import gc; gc.collect()


def test_loss(model):
    """Test masked reconstruction loss."""
    import torch
    from lib.convnext_autoencoder import masked_reconstruction_loss

    print(f"\n--- Masked loss test ---")
    model.eval().float()

    B, H, W = 1, 128, 128
    current = torch.randn(B, 3, H, W)
    target = current.clone()  # Perfect reconstruction target

    with torch.no_grad():
        output, mask = model(current, mask_ratio=0.5)

    loss = masked_reconstruction_loss(output, target, mask)
    print(f"  Masked loss (random weights): {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"
    print("  Loss test passed")


def test_mask_generation(model):
    """Verify mask properties."""
    import torch

    print(f"\n--- Mask generation test ---")
    B, H, W = 4, 256, 256
    ps = model.patch_size

    for ratio in [0.0, 0.2, 0.5, 0.8, 1.0]:
        mask = model.make_mask(B, H, W, ratio, device=torch.device("cpu"))
        actual_ratio = mask.mean().item()
        ph, pw = H // ps, W // ps
        expected_masked = int(ph * pw * ratio)
        actual_masked = mask[0, 0, ::ps, ::ps].sum().item()
        print(f"  ratio={ratio:.1f}: actual={actual_ratio:.3f}, "
              f"patches masked={actual_masked:.0f}/{ph*pw}")
        # Check mask is binary
        assert ((mask == 0) | (mask == 1)).all(), "Mask should be binary"
        # Check approximate ratio (within 1 patch tolerance)
        assert abs(actual_masked - expected_masked) <= 1, \
            f"Patch count off: {actual_masked} vs {expected_masked}"

    print("  Mask generation tests passed")


def main():
    parser = argparse.ArgumentParser(description="Test ConvNeXt-V2 Autoencoder")
    parser.add_argument("--variant", default="atto", choices=["atto", "femto", "pico", "nano", "tiny"])
    parser.add_argument("--no-pretrained", action="store_true", help="Skip pretrained weight loading")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU tests")
    parser.add_argument("--gpu-height", type=int, default=1080)
    parser.add_argument("--gpu-width", type=int, default=1920)
    args = parser.parse_args()

    model = test_construction(args.variant, pretrained=not args.no_pretrained)
    test_mask_generation(model)
    test_forward_cpu(model, height=256, width=256)
    test_loss(model)

    if not args.skip_gpu:
        test_forward_gpu(model, height=args.gpu_height, width=args.gpu_width)

    print(f"\n{'='*60}")
    print("  All tests passed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
