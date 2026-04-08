import vapoursynth as vs
import os, sys

core = vs.core

SCRIPT_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

input_path = globals().get("input")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                          "nafnet_w32mid4_1088x1920_v3.onnx")

clip = core.bs.VideoSource(source=input_path)
orig_w = clip.width
orig_h = clip.height
matrix = "709"

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=matrix)

TARGET_H = ((orig_h + 15) // 16) * 16
TARGET_W = ((orig_w + 15) // 16) * 16
pad_h = TARGET_H - orig_h
pad_w = TARGET_W - orig_w
if pad_h > 0 or pad_w > 0:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "reference-code", "vs-mlrt", "scripts"))
import vsmlrt

clip = vsmlrt.inference(
    clip,
    network_path=MODEL_PATH,
    overlap=(0, 0),
    tilesize=(TARGET_W, TARGET_H),
    backend=vsmlrt.Backend.ORT_CUDA(
        fp16=True,
        device_id=0,
        num_streams=1,
        use_cuda_graph=True,
    ),
)

if pad_h > 0 or pad_w > 0:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=matrix)
clip.set_output()
