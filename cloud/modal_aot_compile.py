"""
AOT Inductor compilation on Modal — compile torch.compile'd NAFNet into a
standalone shared library (.so) that can run without Python.

This produces the same fused Triton kernels that give 78 fps on RTX 3060,
but packaged as a C-callable .so file that libtorch or a custom VapourSynth
plugin can load directly.

Usage:
  PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_aot_compile.py
"""
import modal
import time

app = modal.App("remaster-aot-compile")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .add_local_file("lib/nafnet_arch.py", remote_path="/root/project/lib/nafnet_arch.py")
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
)

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/mnt/data": vol},
)
def aot_compile():
    import torch
    import sys
    import os
    import gc

    sys.path.insert(0, "/root/project")
    from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\nLoading model...")
    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=4,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )

    weight_path = "/mnt/data/checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model.load_state_dict(state, strict=False)
        del ckpt, state
        gc.collect()
        print(f"Loaded weights from {weight_path}")

    model.eval()
    model = swap_layernorm_for_compile(model)
    model.half().cuda()

    dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)

    # Step 1: torch.compile baseline on A10
    print("\n=== torch.compile (Inductor) baseline ===")
    compiled = torch.compile(model, mode="reduce-overhead")
    for _ in range(5):
        with torch.no_grad():
            _ = compiled(dummy)
    torch.cuda.synchronize()

    N = 50
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = compiled(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  {N/elapsed:.1f} fps ({elapsed/N*1000:.1f} ms/frame)")

    # Step 2: AOT Inductor compilation
    print("\n=== AOT Inductor compilation ===")
    # Need to re-export since compiled model can't be used
    exported = torch.export.export(model, (dummy,), strict=False)

    start = time.perf_counter()
    so_path = torch._inductor.aot_compile(
        exported.module(),
        (dummy,),
    )
    compile_time = time.perf_counter() - start
    so_size = os.path.getsize(so_path) / 1024 / 1024
    print(f"  Compiled in {compile_time:.1f}s")
    print(f"  Output: {so_path} ({so_size:.1f} MB)")

    # Step 3: Test loading and running the AOT compiled model
    print("\n=== AOT Inductor benchmark ===")
    runner = torch._export.aot_load(so_path, device="cuda")

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = runner(dummy)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = runner(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  AOT Inductor: {N/elapsed:.1f} fps ({elapsed/N*1000:.1f} ms/frame)")

    # Step 4: Copy artifact to volume
    output_dir = "/mnt/data/checkpoints/nafnet_w32_mid4/aot/"
    os.makedirs(output_dir, exist_ok=True)

    import shutil
    # The .so and associated files
    so_basename = os.path.basename(so_path)
    so_dir = os.path.dirname(so_path)
    for f in os.listdir(so_dir):
        src = os.path.join(so_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {f} ({os.path.getsize(dst)/1024/1024:.1f} MB)")

    vol.commit()
    print(f"\nArtifacts saved to {output_dir}")
    print("Download with: modal volume get upscale-data checkpoints/nafnet_w32_mid4/aot/")


@app.local_entrypoint()
def main():
    aot_compile.remote()
