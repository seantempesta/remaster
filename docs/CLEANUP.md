# Docs Cleanup Summary

Audit of `docs/` as of 2026-04-07. The project has moved from NAFNet to DRUNet
teacher-student distillation. Many docs still reference the old NAFNet architecture,
old speed numbers, old file paths, and old training scripts.

## Files to Archive (Badly Outdated)

These files are specific to the old NAFNet-based pipeline and no longer reflect
the current architecture or workflow. Move to `docs/archive/`.

| File | Reason |
|------|--------|
| `approaches.md` | References SCUNet at 0.52 fps, NAFNet distillation, FlashVSR, old scripts (`pipelines/denoise_batch.py`, `pipelines/denoise_nafnet.py`). All approaches listed are superseded by DRUNet. |
| `architecture-investigation.md` | Entirely about NAFNet w32_mid4 (5.5 fps, 14.3M params). References `training/train_nafnet.py`, `lib/nafnet_arch.py`, `pipelines/denoise_nafnet.py` -- all superseded. |
| `nafnet-speed-investigation.md` | NAFNet-width64 speed analysis on Modal L4. References old checkpoints (`nafnet_distill/`), old scripts. Fully superseded by DRUNet student at 39 fps. |
| `fast-architecture-research.md` | Research brief for finding 30+ fps architecture. References NAFNet configs, `bench/sweep_architectures.py`, `training/train_nafnet.py`. The answer was DRUNet nc=[16,32,64,128] nb=2, which is now deployed. |
| `experiments-log.md` | Experiments 1-7 are historical (Real-ESRGAN, RAFT, SCUNet, NAFNet speed optimization). Useful as history but references many deleted scripts and old approaches. |
| `quantization-research.md` | NAFNet-width64 quantization research. References old baseline (0.59 fps fp16), old TorchAO approaches. DRUNet student already runs at 55 fps TRT INT8. |
| `quantization-aware-training.md` | QAT research for NAFNet INT8. References old baseline (2 fps fp16). DRUNet student already validated at TRT INT8 without QAT. |

## Files to Update (Partially Outdated)

| File | What Needs Updating |
|------|---------------------|
| `architecture.md` | Directory structure references `pipelines/` (production streaming denoisers), `lib/nafnet_arch.py`, `lib/plainnet_arch.py`, `training/` described as "NAFNet distillation training". Data flow diagram shows old pipeline. Modal section references `modal_denoise.py`, `modal_flashvsr.py`, `modal_runner.py`. Needs full rewrite to reflect current DRUNet architecture, `remaster/` pipeline, and current directory layout. |
| `distillation-guide.md` | Has a "(Historical)" header and note about being outdated, which is good. But still appears in the docs listing without being archived. References `training/train_nafnet.py`, `training/generate_pairs.py`, old data paths (`data/train_pairs/`, 1,032 pairs). Should be moved to archive. |
| `setup.md` | References `requirements.txt`, `torch==2.5.1`, `cu121`. Current project uses PyTorch 2.11.0 with cu126. Needs version updates. |
| `tensorrt-implementation.md` | Written for NAFNet-width64 (0.59 fps baseline). The TRT pipeline section is still relevant for DRUNet but the model references and baselines are wrong. |
| `pruning-plan.md` | References DRUNet pruning from 32.6M to ~1M params. The student model (1.06M) was trained via direct distillation, not pruning. The pruning approach was explored but the distillation path won. Should note this outcome. |
| `prd-vapoursynth-pipeline.md` | References NAFNet w32_mid4 throughout (ONNX path `nafnet_w32mid4_1088x1920.onnx`, engine paths, etc). The VapourSynth pipeline is now implemented in `remaster/` with DRUNet student. Should be updated or marked as historical since the PRD has been largely implemented. |
| `detail-recovery-research.md` | References "NAFNet student" throughout and `training/train_nafnet.py`, `pipelines/denoise_nafnet.py`. Core research findings (FFL, DISTS, perceptual loss) are still valid and were adopted. Could update model references to DRUNet. |
| `gpu-profiling-guide.md` | References NAFNet-specific code (`training/train_nafnet.py`) but the profiling techniques are generic and still useful. Update file references. |
| `realtime-playback-research.md` | References NAFNet at 78 fps raw inference. Findings about Python GIL bottleneck are still valid. Speed numbers need updating to DRUNet. |
| `zero-copy-gpu-pipeline.md` | References NAFNet 78 fps baseline. PyNvVideoCodec research is still relevant. Update model references. |
| `vision.md` | References "78 fps" model speed (NAFNet). Current DRUNet student is 39 fps end-to-end, 52 fps TRT FP16 raw. Core vision is still accurate. |

## Files That Are Current (No Changes Needed)

| File | Status |
|------|--------|
| `training-data-plan.md` | Current. Describes the staged data pipeline with SCUNet GAN targets. Matches CLAUDE.md. |
| `modal-graceful-shutdown.md` | Current. Generic Modal infrastructure research, not model-specific. |
| `temporal-consistency-research.md` | Current. Forward-looking research (FastDVDNet, etc). Not model-specific. |
| `research/` subdirectory | Current. Forward-looking temporal research. |

## New Files Added

| File | Description |
|------|-------------|
| `model-card-student.md` | HuggingFace-compatible model card for the 1.06M student |
| `model-card-teacher.md` | HuggingFace-compatible model card for the 32.6M teacher |
| `CLEANUP.md` | This file |

## Recommended Actions

1. Create `docs/archive/` directory
2. Move the 7 "archive" files there
3. Update the 11 "partially outdated" files (or mark with a historical note at top)
4. Update `setup.md` with current PyTorch/CUDA versions
5. Update `architecture.md` to reflect current project structure
