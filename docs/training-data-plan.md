# Training Data Plan

## Objective

Build a diverse, high-quality training dataset that teaches the model two skills:
1. **HEVC artifact removal** — from existing Firefly compressed/denoised pairs
2. **Detail recovery** — from synthetically degraded high-quality source material

The synthetic degradation is **edge-aware**: sharp areas (faces, textures, edges) get blurred while soft areas (bokeh, sky, gradients) stay untouched. This teaches the model to recover detail where it existed, not hallucinate everywhere.

## Sources

### HEVC Artifact Removal (existing)
| Source | Path | Pairs | Notes |
|--------|------|-------|-------|
| Firefly (14 episodes) | `data/train_pairs/` | 1,224 | Input=compressed, Target=SCUNet denoised |

### Synthetic Detail Recovery (edge-aware blur)
| Source | Path | Episodes | Frames | Sigma | Notes |
|--------|------|----------|--------|-------|-------|
| The Expanse S2 | `E:/plex/tv/The Expanse Season 2...` | 13 | 1,200 | 2-5 | Dark sci-fi, space, interiors |
| One Piece S1 | `E:/plex/tv/One.Piece.2023.S01...` | 8 | 400 | 2-8 mix | Colorful, outdoor, costumes, VFX |
| Dune Part Two | `E:/plex/movies/Dune.Part.Two.2024...` | 1 movie | 300 | 2-8 mix | Desert, fabric, skin, extreme lighting |
| Squid Game S2 | `E:/plex/tv/Squid Game - Season 2/` | 7 | 350 | 2-8 mix | Faces, neon lighting, indoor/outdoor |
| Foundation S3 | `E:/plex/tv/foundation.s03e0*.mkv` | 2 | 150 | 2-8 mix | Sci-fi, architecture, costumes |

### Totals
| Category | Training | Validation (10%) |
|----------|----------|-------------------|
| HEVC (Firefly) | 1,224 | ~122 |
| Synthetic (all sources) | 2,400 | ~240 |
| **Total** | **3,624** | **~362** |

## Degradation Strategy

Edge-aware Gaussian blur using Sobel magnitude map:
- Sharp areas (high edge response) get blurred more
- Soft areas (bokeh, sky, gradients) stay untouched
- Prevents the model from over-sharpening areas that should be soft

Sigma distribution (for new sources):
- 60% moderate: sigma 2-5 (detail recovery)
- 40% heavy: sigma 4-8 (teaches detail generation for heavily degraded content)

The Expanse frames (already generated) use sigma 2-5 uniformly.

Optional per-frame augmentation:
- 15% chance of 2x downscale + upscale (resolution loss simulation)

## Validation Strategy

10% holdout from each source, same degradation, separate directory.
Validation frames are extracted with a different random seed (no overlap with training).

Current validation (63 frames) is too small and only covers Firefly + Expanse.
New validation will cover all sources proportionally.

## Directory Structure

Training pairs go into the combined `data/mixed_pairs/` directory:
```
data/mixed_pairs/
  input/
    hevc_e01_00001.png          # Firefly HEVC
    synth_expanse_S02E01_00000.png   # Expanse
    synth_onepiece_S01E01_00000.png  # One Piece
    synth_dune2_00000.png            # Dune 2
    synth_squidgame_S02E01_00000.png # Squid Game
    synth_foundation_S03E01_00000.png # Foundation
  target/
    (matching filenames)
```

Validation:
```
data/mixed_val/
  input/
    (10% sample from all sources, same naming)
  target/
    (matching)
```

## Key Finding

The teacher trained on mixed data (HEVC + synthetic) generalizes beyond its training:
- Removes compression artifacts (trained task)
- Recovers detail from softness (trained task)
- Denoises grain in high-quality Bluray source (emergent behavior)
- Output often exceeds "ground truth" quality on Expanse frames

This happens because the model learns a general "clean + sharpen" objective rather
than task-specific artifact patterns.

## Scripts

- `tools/extract_synthetic_pairs.py` — frame extraction + edge-aware degradation
  - `--source-dir` — path to video files
  - `--sigma-min/--sigma-max` — blur strength range
  - `--num-frames` — frames per run
  - `--no-skip` — regenerate existing frames
  - `--test` — generate 10 test pairs for visual verification
