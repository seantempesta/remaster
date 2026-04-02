"""Export NAFNet ONNX at full 1088x1920 on Modal (needs more RAM than local)."""
import modal
import os
import sys

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("onnx", "numpy")
    .add_local_file("lib/nafnet_arch.py", remote_path="/root/project/lib/nafnet_arch.py")
    .add_local_file("lib/paths.py", remote_path="/root/project/lib/paths.py")
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
)

app = modal.App("export-onnx", image=image)


@app.function(gpu="A10G", volumes={VOL_MOUNT: vol}, timeout=600, memory=32768)
def export_onnx():
    import torch
    import gc
    sys.path.insert(0, "/root/project")
    from lib.nafnet_arch import NAFNet, swap_layernorm_for_export

    vol.reload()

    ckpt_path = f"{VOL_MOUNT}/checkpoints/nafnet_distill/nafnet_best.pth"
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("params", ckpt.get("model", ckpt))
    del ckpt
    gc.collect()
    model.load_state_dict(state)
    del state
    gc.collect()
    swap_layernorm_for_export(model)
    model.eval()

    H, W = 1088, 1920
    dummy = torch.randn(1, 3, H, W)
    onnx_path = f"{VOL_MOUNT}/checkpoints/nafnet_distill/nafnet_w64_1088x1920.onnx"
    print(f"Exporting ONNX at {H}x{W}...")
    torch.onnx.export(model, dummy, onnx_path, opset_version=17,
                      input_names=["input"], output_names=["output"])
    size_mb = os.path.getsize(onnx_path) / 1024**2
    print(f"Saved: {onnx_path} ({size_mb:.0f}MB)")

    import onnx
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("ONNX validation passed")

    vol.commit()
    return f"OK: {size_mb:.0f}MB"


@app.local_entrypoint()
def main():
    result = export_onnx.remote()
    print(f"Export: {result}")

    # Download
    os.makedirs("checkpoints/nafnet_distill", exist_ok=True)
    local = "checkpoints/nafnet_distill/nafnet_w64_1088x1920.onnx"
    print("Downloading ONNX...")
    with open(local, "wb") as f:
        vol.read_file_into_fileobj("/checkpoints/nafnet_distill/nafnet_w64_1088x1920.onnx", f)
    print(f"Saved: {local} ({os.path.getsize(local) / 1024**2:.0f}MB)")
