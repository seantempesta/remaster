"""
Overnight SCUNet Denoiser — processes a full episode.
Streams frames: read -> denoise -> write to video. No intermediate PNGs.
Safe: monitors VRAM, handles errors gracefully, won't fill disk.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import add_scunet_to_path, resolve_scunet_dir
add_scunet_to_path()
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

import os, time, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as TF
from einops import rearrange
from models.network_scunet import SCUNet as net, WMSA
import subprocess

DEVICE = 'cuda'

# Monkey-patch WMSA with SDPA attention (10% faster, less VRAM)
_orig_wmsa_forward = WMSA.forward
def _sdpa_forward(self, x):
    if self.type != 'W':
        x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
    x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
    h_windows = x.size(1)
    w_windows = x.size(2)
    x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
    B, NW, NP, _ = x.shape
    qkv = self.embedding_layer(x).reshape(B, NW, NP, 3, self.n_heads, self.head_dim)
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
    q = q.reshape(B * NW, NP, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(B * NW, NP, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(B * NW, NP, self.n_heads, self.head_dim).transpose(1, 2)
    rel_bias = self.relative_embedding().unsqueeze(0).expand(B * NW, -1, -1, -1)
    if self.type != 'W':
        attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
        attn_mask = attn_mask.reshape(NW, NP, NP).unsqueeze(1).expand(-1, self.n_heads, -1, -1).repeat(B, 1, 1, 1)
        rel_bias = rel_bias.clone()
        rel_bias.masked_fill_(attn_mask, float("-inf"))
    output = TF.scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)
    output = output.transpose(1, 2).reshape(B, NW, NP, -1)
    output = self.linear(output)
    output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
    if self.type != 'W':
        output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
    return output
WMSA.forward = _sdpa_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--model', type=str, default='scunet_color_real_psnr',
                        choices=['scunet_color_real_psnr', 'scunet_color_real_gan'])
    parser.add_argument('--max-frames', type=int, default=-1, help='Max frames to process (-1 for all)')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_denoised{ext}"

    # Video info
    _, _, fps, total_frames, duration = get_video_info(args.input)
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    print(f"Input: {args.input}")
    print(f"  Duration: {duration:.0f}s, FPS: {fps}, Frames: {total_frames}")
    print(f"Output: {args.output}")

    # Load model
    model_path = str(resolve_scunet_dir() / "model_zoo" / f'{args.model}.pth')
    print(f"\nLoading SCUNet ({args.model})...")
    model = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE).half()
    print(f"  Model VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    # Set up ffmpeg reader and writer
    ffmpeg_bin = get_ffmpeg()
    w, h = 1920, 1080  # will be overridden by probe below

    # Probe actual resolution
    w, h, fps, total_frames_probed, duration = get_video_info(args.input)
    if args.max_frames < 0:
        total_frames = total_frames_probed

    read_cmd = [
        ffmpeg_bin, '-hide_banner', '-loglevel', 'error',
        '-i', args.input,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    ]
    if args.max_frames > 0:
        read_cmd += ['-vframes', str(args.max_frames)]
    read_cmd += ['pipe:1']

    fps_str = f"{fps:.6f}"
    write_cmd = [
        ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', fps_str,
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
        '-rc', 'vbr', '-cq', '20',
        '-pix_fmt', 'p010le',
        '-movflags', '+faststart',
        args.output
    ]

    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=10**8)

    print(f"\nProcessing {w}x{h} @ {fps}fps (fp16 direct inference)...")
    print(f"Estimated time: {total_frames * 1.5 / 60:.0f} minutes")

    frame_bytes = w * h * 3
    start = time.time()
    processed = 0
    errors = 0

    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

            try:
                img_t = torch.from_numpy(frame.transpose(2, 0, 1).copy()).half().unsqueeze(0) / 255.0
                img_t = img_t.to(DEVICE)

                with torch.no_grad():
                    out_t = model(img_t)

                out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                del img_t, out_t

            except RuntimeError as e:
                out = frame
                errors += 1
                if errors <= 3:
                    print(f"  WARNING: frame {processed+1} error: {e}")
                torch.cuda.empty_cache()

            writer.stdin.write(out.tobytes())
            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - start
                fps_actual = processed / elapsed
                eta_min = (total_frames - processed) / fps_actual / 60
                vram = torch.cuda.memory_allocated() / 1024**2
                print(f"  [{processed}/{total_frames}] {fps_actual:.2f} fps, "
                      f"VRAM: {vram:.0f}MB, ETA: {eta_min:.0f}min, "
                      f"errors: {errors}")

    except KeyboardInterrupt:
        print(f"\nInterrupted at frame {processed}")
    finally:
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE: {processed} frames in {elapsed/60:.1f} minutes ({processed/elapsed:.2f} fps)")
    print(f"  Errors: {errors}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
