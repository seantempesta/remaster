# Docs Index

## Top-Level

- [setup.md](setup.md) -- Getting started: environment setup, dependencies, conda
- [model-card-teacher.md](model-card-teacher.md) -- HuggingFace model card for 32.6M DRUNet teacher
- [model-card-student.md](model-card-student.md) -- HuggingFace model card for 1.06M DRUNet student

## Research

Active research topics, each in a self-contained folder.

- [research/temporal-context/](research/temporal-context/) -- Recurrent temporal architecture: bottleneck design, PRD, vision backbone candidates
- [research/raft-alignment/](research/raft-alignment/) -- RAFT optical flow for temporal consistency (explored, deprioritized)
- [research/cpp-pipeline/](research/cpp-pipeline/) -- C++ real-time pipeline PRD (bypass Python GIL)
- [research/training-data/](research/training-data/) -- Staged data pipeline: extraction, SCUNet GAN targets, degradation mixing

## Guides

Reference material that remains useful regardless of model architecture.

- [guides/gpu-profiling-guide.md](guides/gpu-profiling-guide.md) -- CUDA profiling with Nsight, torch.profiler, bottleneck analysis
- [guides/modal-graceful-shutdown.md](guides/modal-graceful-shutdown.md) -- Graceful checkpoint saving on Modal spot preemption

## Archive

Outdated docs from the NAFNet era and earlier experiments. Kept for historical
reference but no longer reflect the current DRUNet-based pipeline.

See [archive/](archive/) for the full list.
