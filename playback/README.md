# Real-Time NAFNet Playback with mpv

Enhance video in real-time during playback using NAFNet + TensorRT via mpv + VapourSynth.

## Prerequisites

- Windows 11 with NVIDIA GPU (RTX 3060 or newer)
- NVIDIA drivers 535+ (for TensorRT compatibility)
- NAFNet ONNX model at `checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx`

## Install

### 1. VapourSynth (Portable)

Download the portable release (no install needed):
https://github.com/vapoursynth/vapoursynth/releases

Extract to e.g. `C:\tools\vapoursynth\`. Add to PATH.

### 2. vs-mlrt (TensorRT backend)

Download the latest Windows release:
https://github.com/AmusementClub/vs-mlrt/releases

Extract and copy the plugin DLLs to VapourSynth's plugins directory:
- `vstrt.dll` (TensorRT backend)
- Required TensorRT/CUDA runtime DLLs (included in release)

### 3. mpv

Download mpv for Windows:
https://mpv.io/installation/

Or use the shinchiro builds: https://github.com/shinchiro/mpv-winbuild-cmake/releases

### 4. mpv Configuration

Create/edit `%APPDATA%/mpv/mpv.conf`:

```
# Hardware decode with copy-back (required for VapourSynth)
hwdec=auto-copy

# NAFNet enhancement filter
vf=vapoursynth="C:/Users/sean/src/upscale-experiment/playback/enhance.vpy"

# Optional: show stats overlay with 'i' key
script-opts=stats-font_size=8
```

To toggle enhancement on/off during playback, use a keybinding in
`%APPDATA%/mpv/input.conf`:

```
# Ctrl+E to toggle NAFNet enhancement
Ctrl+e vf toggle vapoursynth="C:/Users/sean/src/upscale-experiment/playback/enhance.vpy"
```

## Usage

```
mpv "path/to/video.mkv"
```

First playback with a new model will take 2-5 minutes to build the TensorRT engine.
The engine is cached for subsequent plays.

Press `Ctrl+E` to toggle enhancement on/off and compare.

## Troubleshooting

- **Black screen / crash on start:** Check that VapourSynth and vs-mlrt DLLs are on
  PATH or in mpv's directory. Run `vspipe enhance.vpy -` to test outside mpv.
- **Slow first play:** TensorRT engine build is one-time. Subsequent plays are instant.
- **Out of VRAM:** Close other GPU apps. Model needs ~1.5-2GB VRAM.
- **Wrong colors:** Ensure the matrix_in_s matches your content (709 for HD, 170m for SD).

## Batch Encoding with VapourSynth

You can also use VapourSynth + vs-mlrt for batch encoding (faster than the Python pipeline):

```
vspipe enhance.vpy --arg "input=episode.mkv" -c y4m - | ffmpeg -i - -c:v hevc_nvenc -preset p4 -cq 20 output.mkv
```
