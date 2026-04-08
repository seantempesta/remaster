# Vision Backbone Candidates for Semantic Embedding

Research into permissive-license vision foundation models for the temporal bottleneck architecture.

## Top Candidates

| Model | Params | Stride | Feature Dim | License | Type | 1080p Native | timm |
|-------|--------|--------|------------|---------|------|-------------|------|
| **ConvNeXt-Tiny IN-22k** | 28.6M | 4/8/16/32 | 96/192/384/768 | **MIT** | CNN, supervised | Yes | `convnext_tiny.fb_in22k` |
| DINOv2 ViT-S/14 | 21M | 14 | 384 | Apache 2.0 | ViT, self-supervised | No (quadratic) | `vit_small_patch14_dinov2.lvd142m` |
| OpenCLIP ConvNeXt-Base | 88M | 4/8/16/32 | 128-1024 | MIT | CNN, contrastive | Yes | partial |
| EVA-02 Small | 21.6M | 14 | 384 | MIT | ViT, MIM | No (quadratic) | `eva02_small_patch14_336` |

## Recommendation

**Primary: `convnext_tiny.fb_in22k`** (MIT, 28.6M params, pure CNN)
- Stage 3 at stride 16 = 67x120x384 at 1080p -- matches our bottleneck resolution
- Pure CNN: linear memory, TRT-compatible, no attention blowup at high res
- ImageNet-22k (14M images, 21K classes) -- broad semantic understanding
- Estimated 33-50 fps at 1080p on RTX 3060

**Secondary: DINOv2 ViT-S/14** (Apache 2.0, 21M params) at reduced resolution (518x518 input, upsample features). Best feature quality but needs resize workaround for 1080p.

**WARNING:** ConvNeXt-V2 weights are CC-BY-NC-4.0 (non-commercial). Only V1 is MIT.

## Rejected

| Model | Reason |
|-------|--------|
| DINOv3 | Custom Meta license, requires permission |
| MoCo v3 | CC-BY-NC-4.0 |
| ConvNeXt-V2 | CC-BY-NC-4.0 |
| MAE weights | CC-BY-NC-4.0 |
| Depth Anything V2 | CC-BY-NC-4.0 (except Small) |

## Usage

```python
import timm
model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, features_only=True)
model.eval().cuda().half()
# features[2] = stride-16: [B, 384, 67, 120] at 1080p
```

See full research details in the agent output.
