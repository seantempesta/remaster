"""Compare model inference across backends to catch quality regressions.

Run: pytest tests/test_inference_quality.py -v -s

All tests run sequentially (single GPU with 6GB VRAM).
"""
import sys
import math
from pathlib import Path

import pytest

# GPU tests must run sequentially (6GB VRAM). Use: pytest -p no:xdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "drunet_student"
CHECKPOINT = CHECKPOINT_DIR / "final.pth"
TRT_FP16_ENGINES = [CHECKPOINT_DIR / "drunet_student_1080p_fp16.engine",
                     CHECKPOINT_DIR / "drunet_student_1080p_fp16_from_fp32.engine"]
TRT_INT8_ENGINE = CHECKPOINT_DIR / "drunet_student_1080p_int8.engine"
ONNX_FP16 = CHECKPOINT_DIR / "drunet_student.onnx"
ONNX_FP32 = CHECKPOINT_DIR / "drunet_student_fp32.onnx"
ORIGINALS_DIR = PROJECT_ROOT / "data" / "originals"
PAD_MULTIPLE = 8


def _psnr(a, b):
    """PSNR between two float tensors in [0,1]."""
    import torch
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 100.0 if mse < 1e-12 else -10.0 * math.log10(mse)


def _laplacian_variance(tensor):
    """Compute Laplacian variance (sharpness) of a [1,C,H,W] tensor."""
    import torch; import torch.nn.functional as F
    gray = tensor.float().mean(dim=1, keepdim=True)
    kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                          dtype=torch.float32, device=tensor.device).reshape(1,1,3,3)
    return F.conv2d(gray, kernel, padding=1).var().item()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_fp32(project_root):
    """Load DRUNet student in FP32 on CUDA."""
    import torch
    from lib.paths import add_kair_to_path
    add_kair_to_path()
    from models.network_unet import UNetRes

    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt))

    model = UNetRes(in_nc=3, out_nc=3, nc=[16, 32, 64, 128], nb=2,
                    act_mode="R", bias=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    for p in model.parameters():
        p.requires_grad = False
    return model


@pytest.fixture(scope="session")
def test_frame():
    """Load first PNG from data/originals/ as [1,3,H,W] FP32 CUDA, padded to 8."""
    import torch
    from PIL import Image
    import numpy as np

    pngs = sorted(ORIGINALS_DIR.glob("*.png"))
    assert len(pngs) > 0, f"No PNGs found in {ORIGINALS_DIR}"
    img = Image.open(pngs[0]).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    # Pad to multiple of 8
    _, _, h, w = tensor.shape
    ph, pw = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE, (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    if ph or pw:
        tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    print(f"Test frame: {pngs[0].name}  shape={list(tensor.shape)}")
    return tensor.cuda()


@pytest.fixture(scope="session")
def test_frame_fp16(test_frame):
    return test_frame.half()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPyTorchPrecision:
    def test_pytorch_fp32_vs_fp16(self, model_fp32, test_frame, test_frame_fp16):
        """FP32 and FP16 inference should produce nearly identical results."""
        import torch

        with torch.no_grad():
            out_fp32 = model_fp32(test_frame)
            model_fp16 = model_fp32.half()
            out_fp16 = model_fp16(test_frame_fp16).float()
            model_fp32.float()  # restore

        psnr = _psnr(out_fp32, out_fp16)
        mean_diff = (out_fp32.mean() - out_fp16.mean()).abs().item()

        print(f"  FP32 vs FP16: PSNR={psnr:.1f} dB, mean_diff={mean_diff:.6f}")
        assert psnr > 60.0, f"PSNR too low: {psnr:.1f} dB (expected >60)"
        assert mean_diff < 0.001, f"Mean diff too large: {mean_diff:.6f}"

    def test_sharpness_preserved(self, model_fp32, test_frame):
        """Model output should retain reasonable sharpness."""
        import torch

        with torch.no_grad():
            output = model_fp32(test_frame)

        input_sharpness = _laplacian_variance(test_frame)
        output_sharpness = _laplacian_variance(output)
        ratio = output_sharpness / max(input_sharpness, 1e-9)

        print(f"  Input sharpness:  {input_sharpness:.4f}")
        print(f"  Output sharpness: {output_sharpness:.4f}")
        print(f"  Ratio: {ratio:.2%}")

        # Sharpness values are small because input is [0,1] not [0,255].
        # The ratio is the meaningful metric -- model should preserve structure.
        assert ratio > 0.4, (
            f"Sharpness ratio too low: {ratio:.2%} (expected >40%)"
        )


class TestTRTEngines:
    @staticmethod
    def _load_and_run_engine(engine_path, input_tensor):
        """Load a TRT engine and run inference.

        Pads input to match engine's static dimensions, crops output back.
        Returns None if engine is incompatible.
        """
        import torch; import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(str(engine_path), "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None  # version mismatch or other deserialization failure
        ctx = engine.create_execution_context()
        # Find input/output tensor names
        in_name, out_name = None, None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                in_name = name
            else:
                out_name = name
        assert in_name and out_name, "Could not find input/output tensors"

        # Engine has static shape -- pad input to match
        engine_shape = list(engine.get_tensor_shape(in_name))  # [1, 3, H, W]
        _, _, orig_h, orig_w = input_tensor.shape
        eng_h, eng_w = engine_shape[2], engine_shape[3]

        if orig_h != eng_h or orig_w != eng_w:
            padded = torch.zeros(1, 3, eng_h, eng_w,
                                 dtype=input_tensor.dtype, device=input_tensor.device)
            padded[:, :, :orig_h, :orig_w] = input_tensor
            run_input = padded
        else:
            run_input = input_tensor.contiguous()

        output = torch.zeros(engine_shape, dtype=torch.float16, device="cuda")
        ctx.set_tensor_address(in_name, run_input.data_ptr())
        ctx.set_tensor_address(out_name, output.data_ptr())

        stream = torch.cuda.Stream()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        # Crop back to original dimensions
        return output[:, :, :orig_h, :orig_w]

    @pytest.mark.parametrize("engine_path", TRT_FP16_ENGINES, ids=lambda p: p.stem)
    def test_trt_fp16_vs_pytorch(self, engine_path, model_fp32, test_frame,
                                  test_frame_fp16):
        """TRT FP16 engine output should closely match PyTorch FP32."""
        import torch
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            pytest.skip("tensorrt not installed")

        if not engine_path.exists():
            pytest.skip(f"Engine not found: {engine_path.name}")

        with torch.no_grad():
            ref = model_fp32(test_frame)

        trt_out = self._load_and_run_engine(engine_path, test_frame_fp16)
        if trt_out is None:
            pytest.skip(f"Engine incompatible with installed TRT version")
        psnr = _psnr(ref, trt_out)

        print(f"  {engine_path.stem} vs PyTorch FP32: PSNR={psnr:.1f} dB")
        assert psnr > 35.0, f"PSNR too low for FP16 engine: {psnr:.1f} dB"

    def test_trt_int8_vs_pytorch(self, model_fp32, test_frame, test_frame_fp16):
        """TRT INT8 engine output should be within acceptable range of PyTorch FP32."""
        import torch
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            pytest.skip("tensorrt not installed")

        if not TRT_INT8_ENGINE.exists():
            pytest.skip(f"Engine not found: {TRT_INT8_ENGINE.name}")

        with torch.no_grad():
            ref = model_fp32(test_frame)

        trt_out = self._load_and_run_engine(TRT_INT8_ENGINE, test_frame_fp16)
        if trt_out is None:
            pytest.skip(f"Engine incompatible with installed TRT version")
        psnr = _psnr(ref, trt_out)

        print(f"  INT8 vs PyTorch FP32: PSNR={psnr:.1f} dB")
        assert psnr > 30.0, f"PSNR too low for INT8 engine: {psnr:.1f} dB"


class TestColorRoundtrip:
    def test_color_roundtrip_10bit(self):
        """RGB -> P010 YUV -> RGB should round-trip within 1/1023 for pure colors."""
        # BT.709 coefficients (same as color_kernels.cu)
        Kr, Kg, Kb = 0.2126, 0.7152, 0.0722

        test_colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "red":   (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue":  (0.0, 0.0, 1.0),
            "gray":  (0.5, 0.5, 0.5),
        }

        tol = 2.0 / 1023.0  # 2 LSB -- accounts for quantize-then-dequantize rounding

        for name, (r, g, b) in test_colors.items():
            # Forward: RGB [0,1] -> 10-bit YUV
            # Y in [64, 940], range 876
            y_val = (Kr * r + Kg * g + Kb * b) * 876.0 + 64.0

            # U,V in [64, 960], range 896
            wg = 1.0 - Kr - Kb  # == Kg
            u_val = (-Kr / 1.8556 * r - wg / 1.8556 * g + 0.5 * b) * 896.0 + 512.0
            v_val = (0.5 * r - wg / 1.5748 * g - Kb / 1.5748 * b) * 896.0 + 512.0

            # Quantize to 10-bit integers (like the GPU would)
            y10 = min(max(round(y_val), 0), 1023)
            u10 = min(max(round(u_val), 0), 1023)
            v10 = min(max(round(v_val), 0), 1023)

            # Inverse: 10-bit YUV -> RGB [0,1]
            y0 = y10 - 64.0
            u0 = u10 - 512.0
            v0 = v10 - 512.0
            scale_y = 1.0 / 876.0
            scale_uv = 1.0 / 896.0

            r2 = y0 * scale_y + 1.5748 * v0 * scale_uv
            g2 = y0 * scale_y - 0.1873 * u0 * scale_uv - 0.4681 * v0 * scale_uv
            b2 = y0 * scale_y + 1.8556 * u0 * scale_uv

            r2 = min(max(r2, 0.0), 1.0)
            g2 = min(max(g2, 0.0), 1.0)
            b2 = min(max(b2, 0.0), 1.0)

            err_r = abs(r2 - r)
            err_g = abs(g2 - g)
            err_b = abs(b2 - b)
            max_err = max(err_r, err_g, err_b)

            print(f"  {name:6s}: ({r},{g},{b}) -> Y={y10} U={u10} V={v10} "
                  f"-> ({r2:.4f},{g2:.4f},{b2:.4f})  max_err={max_err:.6f}")

            assert max_err < tol, (
                f"Color {name} round-trip error {max_err:.6f} exceeds "
                f"10-bit tolerance {tol:.6f}"
            )


class TestONNXPrecision:
    def test_onnx_fp16_vs_fp32(self, test_frame):
        """FP16 and FP32 ONNX models should produce nearly identical results."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        if not ONNX_FP16.exists():
            pytest.skip(f"ONNX FP16 not found: {ONNX_FP16.name}")
        if not ONNX_FP32.exists():
            pytest.skip(f"ONNX FP32 not found: {ONNX_FP32.name}")

        import numpy as np

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess_fp16 = ort.InferenceSession(str(ONNX_FP16), providers=providers)
        sess_fp32 = ort.InferenceSession(str(ONNX_FP32), providers=providers)

        input_fp32 = test_frame.cpu().numpy().astype(np.float32)
        input_fp16 = test_frame.cpu().numpy().astype(np.float16)
        input_name = sess_fp32.get_inputs()[0].name

        out_fp32 = sess_fp32.run(None, {input_name: input_fp32})[0]
        out_fp16 = sess_fp16.run(None, {input_name: input_fp16})[0]

        mse = np.mean((out_fp32.astype(np.float32) - out_fp16.astype(np.float32)) ** 2)
        if mse < 1e-12:
            psnr = 100.0
        else:
            psnr = -10.0 * math.log10(mse)

        mean_diff = abs(out_fp32.mean() - out_fp16.mean())

        print(f"  ONNX FP32 vs FP16: PSNR={psnr:.1f} dB, mean_diff={mean_diff:.6f}")
        assert psnr > 60.0, f"ONNX PSNR too low: {psnr:.1f} dB (expected >60)"
