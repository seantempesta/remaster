"""
Build TensorRT engine on Modal (A10G, 24 GB VRAM, SM 8.6 Ampere).

The RTX 3060 Laptop (6 GB) doesn't have enough VRAM for TRT tactic search
during compilation (needs 5.3 GB workspace). We compile on a larger GPU with
the same SM 8.6 architecture, then download the engine to run locally.

Usage:
  PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_build_trt.py

The engine file will be saved to checkpoints/nafnet_w32_mid4/engines/
"""
import modal
import time

app = modal.App("remaster-build-trt")

# Image with PyTorch + torch_tensorrt + our model code
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        "torch_tensorrt",
        "onnx",
        "onnxscript",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .add_local_file("lib/nafnet_arch.py", remote_path="/root/project/lib/nafnet_arch.py")
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
)

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # SM 8.6 Ampere, 24 GB — same arch as RTX 3060
    timeout=1800,
    volumes={"/mnt/data": vol},
)
def build_engine():
    import torch
    import torch_tensorrt
    import sys
    import os
    import gc

    sys.path.insert(0, "/root/project")
    from lib.nafnet_arch import NAFNet, swap_layernorm_for_export

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")

    # Load model
    print("\nLoading model...")
    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=4,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )

    # Try to load weights from volume
    weight_path = "/mnt/data/checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model.load_state_dict(state, strict=False)
        del ckpt, state
        gc.collect()
        print(f"Loaded weights from {weight_path}")
    else:
        print("WARNING: No weights found, using random weights (engine structure will be correct)")

    model.eval()
    model = swap_layernorm_for_export(model)
    model.half().cuda()

    dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)

    # Approach 1: torch_tensorrt.compile (direct, no ONNX)
    print("\n=== Approach 1: torch_tensorrt.compile (direct FX→TRT) ===")
    start = time.perf_counter()
    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(shape=[1, 3, 1088, 1920], dtype=torch.float16)],
            use_explicit_typing=True,
            workspace_size=8 << 30,  # 8 GB workspace — plenty on A10G
            min_block_size=1,
        )
        compile_time = time.perf_counter() - start
        print(f"Compiled in {compile_time:.1f}s")

        # Warmup
        for _ in range(5):
            _ = trt_model(dummy)
        torch.cuda.synchronize()

        # Benchmark
        N = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N):
                _ = trt_model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fps = N / elapsed
        print(f"torch_tensorrt direct: {fps:.1f} fps ({elapsed/N*1000:.1f} ms/frame)")

        # Save the engine
        engine_path = "/mnt/data/checkpoints/nafnet_w32_mid4/engines/nafnet_w32mid4_trt_sm86.ts"
        torch_tensorrt.save(trt_model, engine_path, output_format="torchscript",
                            inputs=[dummy])
        print(f"Saved TorchScript+TRT to {engine_path}")

    except Exception as e:
        compile_time = time.perf_counter() - start
        print(f"FAILED after {compile_time:.1f}s: {e}")

    # Approach 2: ONNX export + trtexec (for comparison)
    print("\n=== Approach 2: ONNX → trtexec (for comparison) ===")
    onnx_path = "/tmp/nafnet_w32mid4.onnx"
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                          "output": {0: "batch", 2: "height", 3: "width"}},
        )
        # Merge external data
        import onnx
        m = onnx.load(onnx_path, load_external_data=True)
        for init in m.graph.initializer:
            init.ClearField("data_location")
        onnx.save(m, onnx_path)
        del m

        ops = set()
        m2 = onnx.load(onnx_path)
        for n in m2.graph.node:
            ops.add(n.op_type)
        print(f"ONNX ops: {sorted(ops)}")
        print(f"ONNX size: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")
        del m2

    except Exception as e:
        print(f"ONNX export failed: {e}")

    # Approach 3: torch.compile inductor baseline
    print("\n=== Approach 3: torch.compile (inductor baseline) ===")
    model2 = NAFNet(
        img_channel=3, width=32, middle_blk_num=4,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )
    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model2.load_state_dict(state, strict=False)
        del ckpt, state; gc.collect()

    from lib.nafnet_arch import swap_layernorm_for_compile
    model2.eval()
    model2 = swap_layernorm_for_compile(model2)
    model2.half().cuda()
    model2 = torch.compile(model2, mode="reduce-overhead")

    # Warmup (triggers compilation)
    for _ in range(5):
        with torch.no_grad():
            _ = model2(dummy)
    torch.cuda.synchronize()

    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = model2(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = N / elapsed
    print(f"torch.compile (inductor): {fps:.1f} fps ({elapsed/N*1000:.1f} ms/frame)")

    vol.commit()
    print("\nDone. Check /mnt/data/checkpoints/nafnet_w32_mid4/engines/ for output.")


@app.local_entrypoint()
def main():
    build_engine.remote()
