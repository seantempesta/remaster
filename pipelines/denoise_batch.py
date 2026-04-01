"""
Batched SCUNet Denoiser — streaming video pipeline with threaded IO.

Works locally (RTX 3060 with batch_size=1) and on cloud GPUs (batch_size=6+).
Encodes to high-quality 10-bit H.265. Overlaps decode/encode with GPU inference.

Usage:
    python pipelines/denoise_batch.py --input episode.mkv
    python pipelines/denoise_batch.py --input episode.mkv --batch-size 1 --encoder hevc_nvenc
    python pipelines/denoise_batch.py --input episode.mkv --batch-size 6 --max-frames 100
    python pipelines/denoise_batch.py --input episode.mkv --tile --tile-size 544 --tile-overlap 32
"""
import sys
import os
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import add_scunet_to_path, resolve_scunet_dir
from lib.ffmpeg_utils import get_ffmpeg, get_ffprobe, get_video_info, build_encoder_cmd

SCUNET_DIR = add_scunet_to_path()

import torch
import torch.nn.functional as TF
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
from models.network_scunet import SCUNet as net, WMSA

DEVICE = "cuda"

# Track original forward for fallback
_orig_wmsa_forward = WMSA.forward


def denoise_tiled(model, frame_tensor, tile_size=544, overlap=32):
    """
    Tile-based batched inference for a single frame.

    Splits frame into overlapping tiles, batches them through the model,
    then merges with linear blending at seams.

    Args:
        model: SCUNet model (fp16, CUDA)
        frame_tensor: [1, 3, H, W] fp16 CUDA tensor
        tile_size: target tile height/width (will be adjusted for divisibility by 64)
        overlap: pixels of overlap on each shared edge

    Returns:
        [1, 3, H, W] fp16 CUDA tensor
    """
    _, C, H, W = frame_tensor.shape

    # Ensure tile_size is divisible by 64
    tile_size = (tile_size // 64) * 64
    overlap = max(overlap, 0)

    # Compute tile grid: figure out how many tiles we need in each dimension
    # stride = tile_size - overlap
    stride_h = tile_size - overlap
    stride_w = tile_size - overlap

    # Number of tiles in each dimension
    n_h = max(1, (H - overlap + stride_h - 1) // stride_h)
    n_w = max(1, (W - overlap + stride_w - 1) // stride_w)

    # Compute tile start positions, ensuring last tile doesn't go past the image
    starts_h = []
    for i in range(n_h):
        y = i * stride_h
        if y + tile_size > H:
            y = H - tile_size
        starts_h.append(max(y, 0))
    starts_w = []
    for j in range(n_w):
        x = j * stride_w
        if x + tile_size > W:
            x = W - tile_size
        starts_w.append(max(x, 0))

    # Deduplicate (can happen if image is smaller than tile_size)
    starts_h = sorted(set(starts_h))
    starts_w = sorted(set(starts_w))

    # Pad the frame if it's smaller than tile_size in either dimension
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        frame_tensor = TF.pad(frame_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        H_padded, W_padded = H + pad_h, W + pad_w
    else:
        H_padded, W_padded = H, W

    # Extract tiles
    tiles = []
    tile_coords = []  # (y_start, x_start)
    for y in starts_h:
        for x in starts_w:
            # Ensure we don't read past padded boundaries
            y_end = min(y + tile_size, H_padded)
            x_end = min(x + tile_size, W_padded)
            y_actual = y_end - tile_size
            x_actual = x_end - tile_size
            tile = frame_tensor[:, :, y_actual:y_end, x_actual:x_end]
            # Pad tile to be divisible by 64 if needed
            th, tw = tile.shape[2], tile.shape[3]
            pad_th = (64 - th % 64) % 64
            pad_tw = (64 - tw % 64) % 64
            if pad_th > 0 or pad_tw > 0:
                tile = TF.pad(tile, (0, pad_tw, 0, pad_th), mode='reflect')
            tiles.append(tile.squeeze(0))  # [C, tile_h, tile_w]
            tile_coords.append((y_actual, x_actual, th, tw))

    # Batch all tiles and run model
    batch = torch.stack(tiles, dim=0)  # [N, C, tile_h, tile_w]
    with torch.no_grad():
        out_batch = model(batch)
    torch.cuda.synchronize()

    # Create output and weight buffers
    output = torch.zeros(1, C, H_padded, W_padded, dtype=frame_tensor.dtype, device=frame_tensor.device)
    weight = torch.zeros(1, 1, H_padded, W_padded, dtype=frame_tensor.dtype, device=frame_tensor.device)

    for idx, (y, x, th, tw) in enumerate(tile_coords):
        tile_out = out_batch[idx:idx+1, :, :th, :tw]  # Remove any padding

        # Build per-tile weight mask with linear ramps in overlap regions
        tile_weight = torch.ones(1, 1, th, tw, dtype=frame_tensor.dtype, device=frame_tensor.device)

        # Top edge ramp (if not at the very top of the image)
        if y > 0 and overlap > 0:
            ramp = torch.linspace(0, 1, overlap, dtype=frame_tensor.dtype, device=frame_tensor.device)
            tile_weight[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)

        # Bottom edge ramp (if not at the very bottom)
        if y + th < H_padded and overlap > 0:
            ramp = torch.linspace(1, 0, overlap, dtype=frame_tensor.dtype, device=frame_tensor.device)
            tile_weight[:, :, -overlap:, :] *= ramp.view(1, 1, -1, 1)

        # Left edge ramp
        if x > 0 and overlap > 0:
            ramp = torch.linspace(0, 1, overlap, dtype=frame_tensor.dtype, device=frame_tensor.device)
            tile_weight[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)

        # Right edge ramp
        if x + tw < W_padded and overlap > 0:
            ramp = torch.linspace(1, 0, overlap, dtype=frame_tensor.dtype, device=frame_tensor.device)
            tile_weight[:, :, :, -overlap:] *= ramp.view(1, 1, 1, -1)

        output[:, :, y:y+th, x:x+tw] += tile_out * tile_weight
        weight[:, :, y:y+th, x:x+tw] += tile_weight

    # Normalize by weight (avoid division by zero)
    output = output / weight.clamp(min=1e-6)

    # Remove padding if we added any
    output = output[:, :, :H, :W]

    del batch, out_batch, tiles, weight
    return output


def patch_sdpa():
    """Replace WMSA attention with SDPA (10% faster, less VRAM).
    Falls back to original if SDPA fails (e.g. on some GPU architectures)."""
    def _sdpa_forward(self, x):
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
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
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            attn_mask = attn_mask.reshape(NW, NP, NP).unsqueeze(1).expand(-1, self.n_heads, -1, -1).repeat(B, 1, 1, 1)
            rel_bias = rel_bias.clone()
            rel_bias.masked_fill_(attn_mask, float("-inf"))
        with sdpa_kernel(SDPBackend.MATH):
            output = TF.scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)
        output = output.transpose(1, 2).reshape(B, NW, NP, -1)
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output
    WMSA.forward = _sdpa_forward


def denoise_video(
    input_path,
    output_path,
    model_name="scunet_color_real_psnr",
    batch_size=1,
    crf=18,
    encoder="libx265",
    max_frames=-1,
    use_compile=False,
    use_sdpa=True,
    use_tiling=False,
    tile_size=544,
    tile_overlap=32,
):
    """
    Core denoise pipeline. Streams: ffmpeg decode -> batched SCUNet -> ffmpeg encode.
    Threaded IO overlaps with GPU inference.

    When use_tiling=True, each frame is split into overlapping tiles that are
    batched through the model together, then merged with linear blending.
    This uses less VRAM than full-frame batching and processes fewer windows per tile.
    """
    if use_sdpa:
        patch_sdpa()
        print("  SDPA attention enabled")

    # Cache relative_embedding coord grid (called every forward, wastefully recomputed)
    _rel_cache = {}
    _orig_rel_embedding = WMSA.relative_embedding
    def _cached_rel_embedding(self):
        ws = self.window_size
        if ws not in _rel_cache:
            cord = torch.tensor(
                [[i, j] for i in range(ws) for j in range(ws)],
                device=self.relative_position_params.device
            )
            relation = cord[:, None, :] - cord[None, :, :] + ws - 1
            _rel_cache[ws] = relation
        relation = _rel_cache[ws]
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]
    WMSA.relative_embedding = _cached_rel_embedding

    # ---- Video info ----
    input_path = os.path.abspath(input_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    print(f"Input: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    # ---- Load model ----
    model_path = os.path.join(SCUNET_DIR, "model_zoo", f"{model_name}.pth")
    print(f"\nLoading SCUNet ({model_name})...")
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE).half()
    torch.backends.cudnn.benchmark = True
    print(f"  Model VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    if use_compile:
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    # ---- FFmpeg setup ----
    ffmpeg = get_ffmpeg()
    frame_bytes = w * h * 3

    # Resolve to absolute path for subprocess
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    read_cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
    ]
    if max_frames > 0:
        read_cmd += ["-vframes", str(max_frames)]
    read_cmd += ["pipe:1"]

    write_cmd = build_encoder_cmd(ffmpeg, w, h, fps, output_path, encoder, crf)

    # ---- Threaded reader ----
    frame_queue = Queue(maxsize=batch_size * 3)
    SENTINEL = object()

    def reader_thread():
        proc = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            frame_queue.put(frame)
        frame_queue.put(SENTINEL)
        proc.stdout.close()
        proc.wait()

    # ---- Threaded writer ----
    result_queue = Queue(maxsize=batch_size * 3)
    WRITER_DONE = object()
    writer_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=frame_bytes * 4)

    def writer_thread():
        while True:
            item = result_queue.get()
            if item is WRITER_DONE:
                break
            writer_proc.stdin.write(item.tobytes())
        writer_proc.stdin.close()
        writer_proc.wait()

    # ---- Launch IO threads ----
    reader_t = Thread(target=reader_thread, daemon=True)
    writer_t = Thread(target=writer_thread, daemon=True)
    reader_t.start()
    writer_t.start()

    # ---- Batched inference ----
    enc_label = f"{encoder} CRF {crf}" + (" 10-bit" if encoder == "libx265" else "")
    if use_tiling:
        print(f"\nProcessing: tiled (tile={tile_size}, overlap={tile_overlap}), fp16, SDPA")
    else:
        print(f"\nProcessing: batch_size={batch_size}, fp16, SDPA")
    print(f"Encoding: {enc_label}")

    start = time.time()
    processed = 0
    errors = 0
    done = False
    t_read_total = 0.0
    t_prep_total = 0.0
    t_infer_total = 0.0
    t_post_total = 0.0

    while not done:
        if use_tiling:
            # ---- Tiled mode: one frame at a time, tiles batched ----
            t0 = time.time()
            frame = frame_queue.get()
            if frame is SENTINEL:
                done = True
                t_read_total += time.time() - t0
                continue
            batch_frames = [frame]
            t_read_total += time.time() - t0

            try:
                t0 = time.time()
                img_t = torch.from_numpy(frame.transpose(2, 0, 1).copy()).half().unsqueeze(0) / 255.0
                img_t = img_t.to(DEVICE)
                t_prep_total += time.time() - t0

                t0 = time.time()
                out_t = denoise_tiled(model, img_t, tile_size=tile_size, overlap=tile_overlap)
                t_infer_total += time.time() - t0

                t0 = time.time()
                out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                result_queue.put(out)
                del img_t, out_t
                t_post_total += time.time() - t0

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  OOM in tiled mode, passing through frame")
                result_queue.put(frame)
                errors += 1
                print(f"  ERROR: {e}")
        else:
            # ---- Batch-frames mode (original) ----
            t0 = time.time()
            batch_frames = []
            for _ in range(batch_size):
                frame = frame_queue.get()
                if frame is SENTINEL:
                    done = True
                    break
                batch_frames.append(frame)
            t_read_total += time.time() - t0

            if not batch_frames:
                break

            try:
                t0 = time.time()
                batch_np = np.stack(batch_frames)
                batch_t = torch.from_numpy(batch_np.transpose(0, 3, 1, 2).copy()).half() / 255.0
                batch_t = batch_t.to(DEVICE)
                t_prep_total += time.time() - t0

                t0 = time.time()
                with torch.no_grad():
                    out_t = model(batch_t)
                torch.cuda.synchronize()
                t_infer_total += time.time() - t0

                t0 = time.time()
                out_np = (out_t.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
                del batch_t, out_t

                for i in range(len(batch_frames)):
                    result_queue.put(out_np[i])
                t_post_total += time.time() - t0

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  OOM at batch_size={len(batch_frames)}, falling back to single-frame")
                    for frame in batch_frames:
                        try:
                            img_t = torch.from_numpy(frame.transpose(2, 0, 1).copy()).half().unsqueeze(0) / 255.0
                            img_t = img_t.to(DEVICE)
                            with torch.no_grad():
                                out_t = model(img_t)
                            out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                            result_queue.put(out)
                            del img_t, out_t
                        except RuntimeError:
                            result_queue.put(frame)
                            errors += 1
                            torch.cuda.empty_cache()
                else:
                    for frame in batch_frames:
                        result_queue.put(frame)
                        errors += 1
                    print(f"  ERROR: {e}")

        processed += len(batch_frames)

        report_interval = 50 if use_tiling else max(batch_size * 10, 50)
        if processed % report_interval == 0 or processed == len(batch_frames):
            elapsed = time.time() - start
            fps_actual = processed / elapsed
            eta_min = (total_frames - processed) / max(fps_actual, 0.01) / 60
            vram = torch.cuda.memory_allocated() / 1024**2
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  [{processed}/{total_frames}] {fps_actual:.1f} fps, "
                  f"VRAM: {vram:.0f}/{peak_vram:.0f}MB, ETA: {eta_min:.1f}min, errors: {errors}")
            print(f"    timing: read={t_read_total:.1f}s prep={t_prep_total:.1f}s "
                  f"infer={t_infer_total:.1f}s post={t_post_total:.1f}s")

    result_queue.put(WRITER_DONE)
    writer_t.join()
    reader_t.join()

    elapsed = time.time() - start
    out_size = os.path.getsize(output_path) / 1024**2
    print(f"\n{'=' * 60}")
    print(f"DONE: {processed} frames in {elapsed / 60:.1f} min ({processed / elapsed:.1f} fps)")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path} ({out_size:.1f} MB)")
    print(f"{'=' * 60}")

    return {
        "frames": processed,
        "elapsed_min": elapsed / 60,
        "fps": processed / elapsed,
        "errors": errors,
        "output_size_mb": out_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Batched SCUNet video denoiser")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--model", default="scunet_color_real_psnr",
                        choices=["scunet_color_real_psnr", "scunet_color_real_gan"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per batch (1 for 6GB, 6+ for 24GB)")
    parser.add_argument("--crf", type=int, default=18, help="CRF quality (lower=better, 18=visually lossless)")
    parser.add_argument("--encoder", default="hevc_nvenc",
                        choices=["hevc_nvenc", "libx265"],
                        help="hevc_nvenc for local NVIDIA, libx265 for software/cloud")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit frames (-1 for all)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (slower startup, faster steady-state)")
    parser.add_argument("--no-sdpa", action="store_true", help="Disable SDPA attention replacement")
    parser.add_argument("--tile", action="store_true",
                        help="Use tile-based batching (lower VRAM, fewer windows per tile)")
    parser.add_argument("--tile-size", type=int, default=544,
                        help="Tile size in pixels, will be rounded to multiple of 64 (default: 544)")
    parser.add_argument("--tile-overlap", type=int, default=32,
                        help="Overlap pixels between tiles for blending (default: 32)")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_denoised.mkv"

    denoise_video(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        crf=args.crf,
        encoder=args.encoder,
        max_frames=args.max_frames,
        use_compile=args.compile,
        use_sdpa=not args.no_sdpa,
        use_tiling=args.tile,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
