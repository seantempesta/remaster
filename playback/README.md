# Real-Time Playback with mpv

Enhance video in real-time during playback using DRUNet + TensorRT via mpv + VapourSynth.

## Prerequisites

- Windows 10/11 or Linux with NVIDIA GPU (RTX 3060 or newer)
- NVIDIA drivers 535+
- DRUNet student ONNX model at `checkpoints/drunet_student/drunet_student.onnx`

## Install

### 1. VapourSynth (Portable)

Download the portable release (no install needed):
https://github.com/vapoursynth/vapoursynth/releases

Extract and add to PATH.

### 2. vs-mlrt (TensorRT backend)

Download the latest Windows release:
https://github.com/AmusementClub/vs-mlrt/releases

Extract and copy the plugin DLLs to VapourSynth's plugins directory:
- `vstrt.dll` (TensorRT backend)
- Required TensorRT/CUDA runtime DLLs (included in release)

### 3. mpv

Download mpv: https://mpv.io/installation/

### 4. mpv Configuration

Create/edit `%APPDATA%/mpv/mpv.conf` (Windows) or `~/.config/mpv/mpv.conf` (Linux):

```
# Hardware decode with copy-back (required for VapourSynth)
hwdec=auto-copy

# Remaster enhancement filter -- update path to your install
vf=vapoursynth="/path/to/remaster/play.vpy"
```

To toggle enhancement on/off during playback, add to `input.conf`:

```
# Ctrl+E to toggle remaster enhancement
Ctrl+e vf toggle vapoursynth="/path/to/remaster/play.vpy"
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
  PATH or in mpv's directory. Run `vspipe play.vpy -` to test outside mpv.
- **Slow first play:** TensorRT engine build is one-time. Subsequent plays are instant.
- **Out of VRAM:** Close other GPU apps. Model needs ~500 MB VRAM.
- **Wrong colors:** Ensure the matrix_in_s matches your content (709 for HD, 170m for SD).
