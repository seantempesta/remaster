"""
Export DRUNet student model to ONNX for VapourSynth + TensorRT inference.

The exported model uses dynamic spatial dimensions, so a single ONNX file
works for any resolution. TensorRT builds resolution-specific engines and
caches them automatically.

Usage:
  python tools/export_onnx.py
  python tools/export_onnx.py --checkpoint checkpoints/drunet_teacher/final.pth \
      --nc-list 64,128,256,512 --nb 4 --output checkpoints/drunet_teacher/drunet_teacher.onnx

Output: checkpoints/drunet_student/drunet_student.onnx (~4MB)

The ONNX model is resolution-agnostic (dynamic H/W). TensorRT will build
and cache engines per-resolution on first use (~1-3 min for 1080p).
"""
import argparse
import gc
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lib.paths import add_kair_to_path


def export(
    checkpoint: str,
    nc_list: list[int],
    nb: int,
    output: str,
    test_h: int = 1080,
    test_w: int = 1920,
    fp16: bool = True,
):
    """Export a DRUNet model to ONNX."""
    import torch
    import numpy as np

    add_kair_to_path()
    from models.network_unet import UNetRes

    # Build model
    model = UNetRes(in_nc=3, out_nc=3, nc=nc_list, nb=nb,
                    act_mode='R', bias=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"DRUNet nc={nc_list} nb={nb}: {params / 1e6:.2f}M params")

    # Load weights
    print(f"Loading: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("params", ckpt.get("model", ckpt))
    del ckpt
    gc.collect()

    model.load_state_dict(state)
    del state
    gc.collect()
    model.eval()

    # FP16 export: model and dummy in half precision so ONNX declares FP16 I/O.
    # This ensures TRT engines built from this ONNX automatically have FP16 I/O,
    # matching the C++ pipeline's __half* buffers without --inputIOFormats overrides.
    if fp16:
        model.half()
        print("Exporting in FP16 (model.half())")

    # Pad test dimensions to factor of 8 (3 downsample levels)
    pad_factor = 8
    H = ((test_h + pad_factor - 1) // pad_factor) * pad_factor
    W = ((test_w + pad_factor - 1) // pad_factor) * pad_factor
    dummy = torch.randn(1, 3, H, W)
    if fp16:
        dummy = dummy.half()

    # Export to ONNX with dynamic spatial dimensions
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"Exporting ONNX (test shape: 1x3x{H}x{W})...")

    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        do_constant_folding=True,
        dynamo=False,  # Force TorchScript exporter; dynamo (default) emits opset 18 IR that TRT miscompiles
    )

    # Merge external data if PyTorch saved weights separately
    data_file = output + ".data"
    if os.path.exists(data_file):
        print("  Merging external data into single ONNX file...")
        import onnx
        m = onnx.load(output, load_external_data=True)
        for tensor in m.graph.initializer:
            tensor.data_location = 0  # DEFAULT (inline)
            del tensor.external_data[:]
        onnx.save_model(m, output)
        os.remove(data_file)
        del m

    size_mb = os.path.getsize(output) / 1024**2
    print(f"Saved: {output} ({size_mb:.1f}MB)")

    # Validate
    import onnx
    m = onnx.load(output)
    onnx.checker.check_model(m)
    print("ONNX validation passed")

    # Check dynamic axes
    inp = m.graph.input[0]
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"Input shape: {dims}")
    del m

    # Quick sanity check with ONNX Runtime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        input_dtype = np.float16 if fp16 else np.float32
        test_input = np.random.randn(1, 3, H, W).astype(input_dtype)
        outputs = sess.run(None, {"input": test_input})
        out_shape = outputs[0].shape
        print(f"ORT sanity check: input {test_input.shape} ({input_dtype.__name__}) -> output {out_shape}")
        assert out_shape == (1, 3, H, W), f"Shape mismatch: {out_shape}"
        print("Sanity check passed")
    except ImportError:
        print("onnxruntime not installed, skipping sanity check")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Export DRUNet model to ONNX for TensorRT inference"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "drunet_student", "final.pth"),
        help="Path to .pth checkpoint (default: drunet_student/final.pth)",
    )
    parser.add_argument(
        "--nc-list", default="16,32,64,128",
        help="Channel widths per level, comma-separated (default: 16,32,64,128)",
    )
    parser.add_argument(
        "--nb", type=int, default=2,
        help="Residual blocks per level (default: 2)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output ONNX path (default: auto from checkpoint dir)",
    )
    parser.add_argument(
        "--test-h", type=int, default=1080,
        help="Test height for export validation (default: 1080)",
    )
    parser.add_argument(
        "--test-w", type=int, default=1920,
        help="Test width for export validation (default: 1920)",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Export in FP32 instead of FP16 (default: FP16)",
    )

    args = parser.parse_args()

    nc_list = [int(x) for x in args.nc_list.split(",")]

    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(ckpt_dir, "drunet_student.onnx")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    export(
        checkpoint=args.checkpoint,
        nc_list=nc_list,
        nb=args.nb,
        output=args.output,
        test_h=args.test_h,
        test_w=args.test_w,
        fp16=not args.fp32,
    )


if __name__ == "__main__":
    main()
