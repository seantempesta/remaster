"""Export NAFNet w32_mid4 to ONNX at 1088x1920 on Modal."""
import modal
import os
import sys

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.11.0", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("onnx", "onnxscript", "onnxruntime", "numpy")
    .add_local_file("lib/nafnet_arch.py", remote_path="/root/project/lib/nafnet_arch.py")
    .add_local_file("lib/paths.py", remote_path="/root/project/lib/paths.py")
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
)

app = modal.App("export-onnx-w32", image=image)


@app.function(gpu="T4", volumes={VOL_MOUNT: vol}, timeout=600, memory=16384)
def export_onnx():
    import torch
    import gc
    sys.path.insert(0, "/root/project")
    from lib.nafnet_arch import NAFNet, swap_layernorm_for_export

    vol.reload()

    ckpt_path = f"{VOL_MOUNT}/checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build w32_mid4 architecture
    model = NAFNet(img_channel=3, width=32, middle_blk_num=4,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("params", ckpt.get("model", ckpt))
    del ckpt; gc.collect()
    model.load_state_dict(state)
    del state; gc.collect()

    # Use export-safe LayerNorm (no custom autograd)
    swap_layernorm_for_export(model)
    model.eval()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"NAFNet w=32 mid=4: {params_m:.1f}M params")

    # Export at padded 1080p (1088x1920, multiples of 16)
    H, W = 1088, 1920
    dummy = torch.randn(1, 3, H, W)

    onnx_dir = f"{VOL_MOUNT}/checkpoints/nafnet_w32_mid4"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = f"{onnx_dir}/nafnet_w32mid4_1088x1920.onnx"

    print(f"Exporting ONNX at {H}x{W}...")
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    # PyTorch 2.11's dynamo exporter may save weights as external data.
    # Merge everything into a single self-contained ONNX file.
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        print("  Merging external data into single ONNX file...")
        m = onnx.load(onnx_path, load_external_data=True)
        # Clear external data references so save_model writes inline
        for tensor in m.graph.initializer:
            tensor.data_location = 0  # DEFAULT (inline)
            del tensor.external_data[:]
        onnx.save_model(m, onnx_path)
        os.remove(data_file)
        del m

    size_mb = os.path.getsize(onnx_path) / 1024**2
    print(f"Saved: {onnx_path} ({size_mb:.0f}MB)")

    # Validate ONNX
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("ONNX validation passed")
    del m

    # Quick sanity check with onnxruntime
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, 3, H, W).astype(np.float32)
    outputs = sess.run(None, {"input": test_input})
    out_shape = outputs[0].shape
    print(f"ORT sanity check: input {test_input.shape} -> output {out_shape}")
    assert out_shape == (1, 3, H, W), f"Shape mismatch: {out_shape}"
    print("Sanity check passed")

    vol.commit()
    return f"OK: {onnx_path} ({size_mb:.0f}MB)"


@app.local_entrypoint()
def main():
    # Upload checkpoint to volume if not already there
    local_ckpt = "checkpoints/nafnet_w32_mid4/nafnet_best.pth"
    if not os.path.exists(local_ckpt):
        print(f"ERROR: {local_ckpt} not found")
        sys.exit(1)

    try:
        with vol.batch_upload() as batch:
            batch.put_file(local_ckpt, "/checkpoints/nafnet_w32_mid4/nafnet_best.pth")
        print(f"Uploaded {os.path.getsize(local_ckpt) / 1024**2:.0f}MB")
    except FileExistsError:
        print("Checkpoint already on volume, skipping upload.")

    # Run export
    result = export_onnx.remote()
    print(f"Export: {result}")

    # Download ONNX
    local_onnx = "checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx"
    print(f"Downloading ONNX to {local_onnx}...")
    with open(local_onnx, "wb") as f:
        vol.read_file_into_fileobj("/checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx", f)
    print(f"Saved: {local_onnx} ({os.path.getsize(local_onnx) / 1024**2:.0f}MB)")
