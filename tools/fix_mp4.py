"""Fix partial mp4 using recover_mp4 approach — rebuild moov from mdat."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess, os, struct
from lib.paths import DATA_DIR
from lib.ffmpeg_utils import get_ffmpeg

ffmpeg = get_ffmpeg()

inp = str(DATA_DIR / 'Firefly_S01E02_denoised - Copy.mp4')
out = str(DATA_DIR / 'Firefly_S01E02_denoised_preview.mp4')

# The issue: ffmpeg writes mp4 with moov at the end. Since the stream is
# still open, the copy has mdat but no moov.
# Solution: use ffmpeg to create a reference mp4 with same codec settings
# from a few frames, then use that moov as a template.

# First, create a short reference clip with same settings
ref = str(DATA_DIR / 'temp_ref.mp4')
mid_clip = str(DATA_DIR / 'clip_mid_scunet.mp4')

# Actually simpler: just read the broken file as a pipe with known params
# The mdat in the file IS valid h264 in mp4 mdat boxes. We need to extract it properly.

# Parse mp4 boxes to find mdat
print("Parsing mp4 boxes...")
with open(inp, 'rb') as f:
    offset = 0
    while True:
        f.seek(offset)
        header = f.read(8)
        if len(header) < 8:
            break
        size = struct.unpack('>I', header[:4])[0]
        box_type = header[4:8].decode('ascii', errors='replace')
        if size == 0:
            break
        if size == 1:  # 64-bit size
            ext = f.read(8)
            size = struct.unpack('>Q', ext)[0]
        print(f"  Box: '{box_type}' at offset {offset}, size {size}")
        if box_type == 'mdat':
            mdat_offset = offset + 8
            mdat_size = size - 8
            print(f"  -> mdat data: offset={mdat_offset}, size={mdat_size/1024**2:.1f}MB")
        offset += size
        if offset > os.path.getsize(inp):
            break

# Extract raw h264 from mdat and wrap properly
print("\nExtracting h264 from mdat...")
raw_h264 = str(DATA_DIR / 'temp_raw.h264')
with open(inp, 'rb') as fi, open(raw_h264, 'wb') as fo:
    fi.seek(mdat_offset)
    remaining = mdat_size
    while remaining > 0:
        chunk = fi.read(min(remaining, 1024*1024))
        if not chunk:
            break
        fo.write(chunk)
        remaining -= len(chunk)

print(f"Extracted {os.path.getsize(raw_h264)/1024**2:.1f}MB raw data")

# Try to mux with explicit codec params matching our encoding settings
r = subprocess.run([
    ffmpeg, '-y',
    '-f', 'h264', '-framerate', '23.976',
    '-i', raw_h264,
    '-c', 'copy', '-movflags', 'faststart',
    out
], capture_output=True, text=True)

# Cleanup
if os.path.exists(raw_h264):
    os.remove(raw_h264)

if r.returncode == 0 and os.path.getsize(out) > 1000:
    size_mb = os.path.getsize(out) / 1024**2
    print(f"\nFixed! {size_mb:.1f}MB — {out}")
else:
    print(f"\nDirect mux failed, trying Annex B conversion...")
    # The mdat contains h264 in AVCC format (length-prefixed NALUs),
    # not Annex B (start-code prefixed). Convert.
    raw_h264 = str(DATA_DIR / 'temp_raw.h264')
    annexb = str(DATA_DIR / 'temp_annexb.h264')

    with open(inp, 'rb') as fi, open(annexb, 'wb') as fo:
        fi.seek(mdat_offset)
        remaining = mdat_size
        buf = fi.read(min(remaining, mdat_size))
        pos = 0
        nals = 0
        while pos < len(buf) - 4:
            nalu_len = struct.unpack('>I', buf[pos:pos+4])[0]
            if nalu_len <= 0 or pos + 4 + nalu_len > len(buf):
                break
            fo.write(b'\x00\x00\x00\x01')
            fo.write(buf[pos+4:pos+4+nalu_len])
            pos += 4 + nalu_len
            nals += 1

    print(f"  Extracted {nals} NAL units")

    r = subprocess.run([
        ffmpeg, '-y',
        '-f', 'h264', '-framerate', '23.976',
        '-i', annexb,
        '-c', 'copy', '-movflags', 'faststart',
        out
    ], capture_output=True, text=True)

    if os.path.exists(annexb):
        os.remove(annexb)

    if r.returncode == 0 and os.path.getsize(out) > 1000:
        size_mb = os.path.getsize(out) / 1024**2
        print(f"Fixed! {size_mb:.1f}MB — {out}")
    else:
        print(f"Still failed: {r.stderr[-300:]}")
