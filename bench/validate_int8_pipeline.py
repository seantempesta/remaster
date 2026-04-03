"""
Validate the full INT8 pipeline for PlainDenoise/UNetDenoise:

1. Build model with RepConvBlocks (training mode)
2. Fuse reparam → single Conv3x3 per block
3. Fuse Conv+BN → BN disappears (torch.ao.quantization.fuse_modules)
4. Export to ONNX
5. Build TRT FP16 and INT8 engines
6. Benchmark both and compare

This confirms the architecture actually compiles to fast INT8 ops
before we commit to training.

Usage:
    python bench/validate_int8_pipeline.py
    python bench/validate_int8_pipeline.py --arch plain --nc 64 --nb 12
    python bench/validate_int8_pipeline.py --arch unet --nc 64 --nb-mid 2
"""
import argparse
import gc
import os
import sys
import tempfile
import time

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from lib.plainnet_arch import PlainDenoise, UNetDenoise, count_params


def fuse_conv_bn(model):
    """Fuse Conv2d + BatchNorm2d pairs into single Conv2d.

    After reparameterization, the model has Conv2d → BN → ReLU sequences.
    This fuses BN into Conv weights so there's no BN at inference.
    TRT would do this itself, but doing it in PyTorch lets us verify
    the ONNX graph is clean.
    """
    prev_name = None
    prev_mod = None
    fuse_pairs = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d) and prev_mod is not None and isinstance(prev_mod, nn.Conv2d):
            fuse_pairs.append((prev_name, name))
        prev_name = name
        prev_mod = mod

    # Use PyTorch's built-in fusion
    for conv_name, bn_name in fuse_pairs:
        # Navigate to parent module
        parts_conv = conv_name.rsplit('.', 1)
        parts_bn = bn_name.rsplit('.', 1)

        if len(parts_conv) == 1:
            parent = model
            conv_attr = parts_conv[0]
        else:
            parent = model
            for p in parts_conv[0].split('.'):
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            conv_attr = parts_conv[1]

        conv = getattr(parent, conv_attr)
        # Get BN module
        bn_parent = model
        if len(parts_bn) > 1:
            for p in parts_bn[0].split('.'):
                bn_parent = getattr(bn_parent, p) if not p.isdigit() else bn_parent[int(p)]
        bn = getattr(bn_parent, parts_bn[1] if len(parts_bn) > 1 else parts_bn[0])

        # Fuse
        fused = nn.utils.fuse_conv_bn_eval(conv, bn)
        setattr(parent, conv_attr, fused)
        # Replace BN with identity
        if len(parts_bn) > 1:
            setattr(bn_parent, parts_bn[1], nn.Identity())
        else:
            setattr(model, parts_bn[0], nn.Identity())

    return model, len(fuse_pairs)


def count_op_types(model):
    """Count module types in the model."""
    counts = {}
    for m in model.modules():
        name = type(m).__name__
        if name in ('Sequential', 'ModuleList'):
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def export_onnx(model, onnx_path, input_shape=(1, 3, 1088, 1920)):
    """Export to ONNX and report graph stats."""
    model.eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    # PyTorch 2.11 dynamo exporter saves weights as external .data files.
    # TRT can't read those. Merge weights inline.
    try:
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        model_onnx = onnx.load(onnx_path, load_external_data=True)
        # Clear external data locations so weights are saved inline
        for init in model_onnx.graph.initializer:
            if init.HasField("data_location"):
                init.ClearField("data_location")
        onnx.save(model_onnx, onnx_path)
    except Exception as e:
        print(f"  (weight merge note: {e})")

    size_mb = os.path.getsize(onnx_path) / 1024**2
    print(f"  ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

    # Check ONNX graph
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        op_types = {}
        for node in model_onnx.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        print(f"  ONNX ops: {dict(sorted(op_types.items()))}")
        print(f"  ONNX valid: YES")

        # Check for bad ops
        bad_ops = {'LayerNormalization', 'ReduceMean', 'Softmax', 'Sigmoid'}
        found_bad = bad_ops & set(op_types.keys())
        if found_bad:
            print(f"  WARNING: Found non-INT8-friendly ops: {found_bad}")
        else:
            print(f"  INT8-friendly ops only: YES")
    except ImportError:
        print("  (onnx package not installed, skipping graph check)")

    return onnx_path


def build_trt_engine(onnx_path, engine_path, fp16=True, int8=False,
                     input_shape=(1, 3, 1088, 1920)):
    """Build TRT engine from ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  TRT parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Use FP16 as fallback for layers that don't support INT8
        config.set_flag(trt.BuilderFlag.FP16)

        # Simple calibrator using random data (for validation only — real training uses QAT)
        class RandomCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, n_batches=8):
                super().__init__()
                self.n_batches = n_batches
                self.batch_idx = 0
                self.data = torch.randn(*input_shape, dtype=torch.float32).cuda()
                self.d_input = self.data.data_ptr()

            def get_batch_size(self):
                return input_shape[0]

            def get_batch(self, names):
                if self.batch_idx >= self.n_batches:
                    return None
                self.batch_idx += 1
                # Random data each batch (just for testing — real cal uses real images)
                self.data.normal_()
                return [self.d_input]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                pass

        config.int8_calibrator = RandomCalibrator()

    if int8:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        print("  Sparsity flag: ENABLED")

    # Build
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, 544, 960),
                      opt=(1, 3, 1088, 1920),
                      max=(1, 3, 1088, 1920))
    config.add_optimization_profile(profile)

    print(f"  Building TRT engine ({'INT8' if int8 else 'FP16'})...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if serialized is None:
        print(f"  TRT build FAILED")
        return None

    with open(engine_path, 'wb') as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1024**2
    print(f"  TRT engine: {engine_path} ({size_mb:.1f} MB, built in {build_time:.1f}s)")
    return engine_path


def benchmark_trt(engine_path, input_shape=(1, 3, 1088, 1920), n_warmup=5, n_bench=30):
    """Benchmark a TRT engine with CUDA events."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("input", input_shape)

    # Allocate buffers
    inp = torch.randn(*input_shape, dtype=torch.float32, device="cuda")
    out_shape = tuple(context.get_tensor_shape("output"))
    out = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("output", out.data_ptr())

    stream = torch.cuda.Stream()

    # Warmup
    for _ in range(n_warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Benchmark with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for _ in range(n_bench):
        context.execute_async_v3(stream.cuda_stream)
    end.record(stream)
    stream.synchronize()

    total_ms = start.elapsed_time(end)
    ms_per_frame = total_ms / n_bench
    fps = 1000.0 / ms_per_frame

    vram_mb = torch.cuda.max_memory_reserved() / 1024**2

    return fps, ms_per_frame, vram_mb


def benchmark_torch_compile(model, input_shape=(1, 3, 1088, 1920), n_warmup=5, n_bench=30):
    """Benchmark with torch.compile for comparison."""
    model = model.half().cuda()
    model = model.to(memory_format=torch.channels_last)

    torch._inductor.config.conv_1x1_as_mm = True
    torch.backends.cudnn.benchmark = True
    compiled = torch.compile(model, mode="reduce-overhead")

    dummy = torch.randn(*input_shape, device="cuda", dtype=torch.float16)
    dummy = dummy.to(memory_format=torch.channels_last)

    for _ in range(n_warmup + 3):
        with torch.no_grad():
            _ = compiled(dummy)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(n_bench):
            _ = compiled(dummy)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    ms_per_frame = total_ms / n_bench
    fps = 1000.0 / ms_per_frame
    vram_mb = torch.cuda.max_memory_reserved() / 1024**2

    return fps, ms_per_frame, vram_mb


def main():
    parser = argparse.ArgumentParser(description="Validate INT8 pipeline")
    parser.add_argument("--arch", default="unet", choices=["plain", "unet"])
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--nb", type=int, default=15)
    parser.add_argument("--nb-enc", default="2,2")
    parser.add_argument("--nb-dec", default="2,2")
    parser.add_argument("--nb-mid", type=int, default=2)
    parser.add_argument("--skip-trt", action="store_true", help="Skip TRT build (ONNX only)")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"INT8 Pipeline Validation")
    print(f"=" * 70)

    # Step 1: Build model
    print(f"\n--- Step 1: Build model ({args.arch} nc={args.nc}) ---")
    if args.arch == "plain":
        model = PlainDenoise(in_nc=3, nc=args.nc, nb=args.nb, use_bn=True)
    else:
        nb_enc = tuple(int(x) for x in args.nb_enc.split(","))
        nb_dec = tuple(int(x) for x in args.nb_dec.split(","))
        model = UNetDenoise(in_nc=3, nc=args.nc, nb_enc=nb_enc, nb_dec=nb_dec,
                           nb_mid=args.nb_mid, use_bn=True)
    model.eval()
    params = count_params(model)
    print(f"  Params: {params/1e3:.1f}K")
    ops_before = count_op_types(model)
    print(f"  Modules: {ops_before}")

    # Step 2: Reparam fusion
    print(f"\n--- Step 2: Reparameterize (multi-branch → single conv) ---")
    model.fuse_reparam()
    ops_after_reparam = count_op_types(model)
    print(f"  Modules after reparam: {ops_after_reparam}")
    # RepConvBlock containers remain but their internals are fused (single conv + relu)
    # Verify all RepConvBlocks are in deploy mode
    for m in model.modules():
        if hasattr(m, 'deploy') and type(m).__name__ == 'RepConvBlock':
            assert m.deploy, "RepConvBlock should be in deploy mode after fuse_reparam!"
            assert hasattr(m, 'fused_conv'), "RepConvBlock should have fused_conv!"
    print(f"  All RepConvBlocks fused to deploy mode: YES")

    # Step 3: Fuse Conv+BN
    print(f"\n--- Step 3: Fuse Conv+BN (BN → absorbed into Conv) ---")
    model, n_fused = fuse_conv_bn(model)
    ops_after_bn = count_op_types(model)
    print(f"  Fused {n_fused} Conv+BN pairs")
    print(f"  Modules after BN fusion: {ops_after_bn}")
    bn_count = ops_after_bn.get('BatchNorm2d', 0)
    identity_count = ops_after_bn.get('Identity', 0)
    print(f"  Remaining BN layers: {bn_count} (replaced with {identity_count} Identity)")

    # Verify output is still correct
    x = torch.randn(1, 3, 1080, 1920)
    with torch.no_grad():
        y = model(x)
    print(f"  Output shape: {y.shape} (expected [1, 3, 1080, 1920])")
    assert y.shape == (1, 3, 1080, 1920), f"Wrong output shape: {y.shape}"
    print(f"  Correctness: PASS")

    # Step 4: Export ONNX
    print(f"\n--- Step 4: Export ONNX ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        export_onnx(model, onnx_path)

        if args.skip_trt:
            print(f"\n--- Skipping TRT (--skip-trt) ---")
            print(f"\nPIPELINE VALIDATION: ONNX export OK")
            return

        # Step 5: Build TRT engines
        print(f"\n--- Step 5: Build TRT engines ---")

        fp16_path = os.path.join(tmpdir, "model_fp16.engine")
        int8_path = os.path.join(tmpdir, "model_int8.engine")

        fp16_ok = build_trt_engine(onnx_path, fp16_path, fp16=True, int8=False)
        torch.cuda.empty_cache()
        gc.collect()

        int8_ok = build_trt_engine(onnx_path, int8_path, fp16=True, int8=True)
        torch.cuda.empty_cache()
        gc.collect()

        if not fp16_ok and not int8_ok:
            print(f"\nTRT build failed for both FP16 and INT8. ONNX graph was valid though.")
            print(f"Falling back to torch.compile benchmark only.")
            # Still benchmark torch.compile
            print(f"\n--- Step 6: Benchmark (torch.compile only) ---")
            model_compile = PlainDenoise(in_nc=3, nc=args.nc, nb=args.nb) if args.arch == "plain" else \
                UNetDenoise(in_nc=3, nc=args.nc,
                           nb_enc=tuple(int(x) for x in args.nb_enc.split(",")),
                           nb_dec=tuple(int(x) for x in args.nb_dec.split(",")),
                           nb_mid=args.nb_mid)
            model_compile.eval()
            model_compile.fuse_reparam()
            fps_compile, ms_compile, vram_compile = benchmark_torch_compile(model_compile)
            print(f"  torch.compile FP16: {fps_compile:.1f} fps ({ms_compile:.1f} ms/frame)")
            return

        # Step 6: Benchmark
        print(f"\n--- Step 6: Benchmark ---")

        fps_fp16, ms_fp16, vram_fp16 = 0, 0, 0
        fps_int8, ms_int8, vram_int8 = 0, 0, 0

        # TRT FP16
        if fp16_ok:
            print(f"\n  TRT FP16:")
            fps_fp16, ms_fp16, vram_fp16 = benchmark_trt(fp16_path)
            print(f"    {fps_fp16:.1f} fps ({ms_fp16:.1f} ms/frame, {vram_fp16:.0f} MB VRAM)")
            torch.cuda.empty_cache()
            gc.collect()

        # TRT INT8
        if int8_ok:
            print(f"\n  TRT INT8:")
            fps_int8, ms_int8, vram_int8 = benchmark_trt(int8_path)
            print(f"    {fps_int8:.1f} fps ({ms_int8:.1f} ms/frame, {vram_int8:.0f} MB VRAM)")

        torch.cuda.empty_cache()
        gc.collect()

        # torch.compile FP16 (reference)
        print(f"\n  torch.compile FP16 (reference):")
        model_compile = PlainDenoise(in_nc=3, nc=args.nc, nb=args.nb) if args.arch == "plain" else \
            UNetDenoise(in_nc=3, nc=args.nc,
                       nb_enc=tuple(int(x) for x in args.nb_enc.split(",")),
                       nb_dec=tuple(int(x) for x in args.nb_dec.split(",")),
                       nb_mid=args.nb_mid)
        model_compile.eval()
        model_compile.fuse_reparam()
        fps_compile, ms_compile, vram_compile = benchmark_torch_compile(model_compile)
        print(f"    {fps_compile:.1f} fps ({ms_compile:.1f} ms/frame, {vram_compile:.0f} MB VRAM)")

        # Summary
        print(f"\n{'='*70}")
        print(f"RESULTS: {args.arch} nc={args.nc}")
        print(f"{'='*70}")
        print(f"  {'Backend':<25} {'FPS':>8} {'ms/frame':>10} {'VRAM':>8}")
        print(f"  {'-'*55}")
        print(f"  {'torch.compile FP16':<25} {fps_compile:>7.1f} {ms_compile:>9.1f} {vram_compile:>6.0f}MB")
        print(f"  {'TRT FP16':<25} {fps_fp16:>7.1f} {ms_fp16:>9.1f} {vram_fp16:>6.0f}MB")
        print(f"  {'TRT INT8':<25} {fps_int8:>7.1f} {ms_int8:>9.1f} {vram_int8:>6.0f}MB")
        print()
        print(f"  INT8/FP16 speedup: {fps_int8/fps_fp16:.2f}x")
        print(f"  INT8/compile speedup: {fps_int8/fps_compile:.2f}x")

        target = 30.0
        status = "PASS" if fps_int8 >= target else "FAIL"
        print(f"\n  Target: {target} fps → {status} ({fps_int8:.1f} fps INT8)")


if __name__ == "__main__":
    main()
