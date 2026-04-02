# GPU Profiling Guide for NAFNet Training

Diagnosing low GPU utilization (~40%) during CNN training on H100/A100.

## Quick Summary of Tools

| Tool | Overhead | What it shows | Best for |
|------|----------|---------------|----------|
| `torch.profiler` | Medium (~10-20%) | Chrome trace, kernel times, memory, CPU/GPU gaps | Full picture, first investigation |
| CUDA Events | Negligible | Wall-clock per phase | Always-on phase timing |
| Nsight Systems (`nsys`) | Low (~5%) | GPU timeline, kernel gaps, CUDA API calls | Kernel launch overhead, concurrency |
| `torch.utils.bottleneck` | High | cProfile + CUDA time | Quick one-shot, not useful mid-training |

## 1. torch.profiler (Chrome Trace)

The best first step. Profiles a few iterations, exports a trace viewable in Chrome (`chrome://tracing`) or Perfetto (https://ui.perfetto.dev).

### Minimal Integration

Drop this into the training loop with zero changes to existing code:

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Profile iterations 10-15 (skip warmup), export chrome trace
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=2,      # skip first 2 iterations (cold start)
        warmup=3,    # warmup 3 iterations (discard data, but cuDNN is warm)
        active=5,    # record 5 iterations
        repeat=1,    # do this once
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_traces"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,      # shows Python call stacks — useful but adds overhead
)
prof.start()

for iteration in range(start_iter, args.max_iters):
    # ... existing training step ...
    
    prof.step()  # tell profiler we finished one iteration
    
    # Stop profiling after enough iterations
    if iteration - start_iter >= 10:
        prof.stop()
        break  # or continue training without profiler
```

### Lighter Version (No TensorBoard, Direct Chrome Trace)

```python
from torch.profiler import profile, record_function, ProfilerActivity

# Profile exactly 5 iterations after warmup
PROFILE_START = 10  # start profiling at this iteration
PROFILE_ITERS = 5

if iteration == PROFILE_START:
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    )
    prof.__enter__()

if PROFILE_START <= iteration < PROFILE_START + PROFILE_ITERS:
    # Annotate phases for the trace
    pass  # (see annotated loop below)

if iteration == PROFILE_START + PROFILE_ITERS:
    prof.__exit__(None, None, None)
    # Export chrome trace
    prof.export_chrome_trace("nafnet_profile.json")
    # Print summary table
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
```

### Annotating the Training Loop

Add `record_function` calls to label phases in the trace:

```python
for iteration in range(start_iter, args.max_iters):
    with record_function("data_loading"):
        try:
            inp_batch, tgt_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inp_batch, tgt_batch = next(data_iter)

    with record_function("to_device"):
        inp_batch = inp_batch.to(device, non_blocking=True)
        tgt_batch = tgt_batch.to(device, non_blocking=True)

    with record_function("nafnet_forward"):
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(inp_batch)
            pixel_loss = criterion(pred, tgt_batch)

    with record_function("vgg_perceptual_forward"):
        if perceptual_criterion is not None:
            with torch.amp.autocast("cuda", enabled=False):
                p_loss = perceptual_criterion(pred.float(), tgt_batch.float())
            loss = pixel_loss + args.perceptual_weight * p_loss
        else:
            loss = pixel_loss

    with record_function("backward"):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

    with record_function("optimizer_step"):
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

### Viewing Results

1. **Perfetto UI** (recommended): Upload the `.json` trace to https://ui.perfetto.dev
2. **Chrome**: Navigate to `chrome://tracing`, click Load, select the `.json` file
3. **TensorBoard**: If using `tensorboard_trace_handler`, run `tensorboard --logdir=./profiler_traces`

In the trace, look for:
- **Gaps between GPU kernels** = CPU overhead (kernel launch, Python, data loading)
- **Long CPU bars with no GPU activity** = DataLoader blocking
- **fp32 kernels that could be fp16** = VGG eating compute time

### Key Averages Table

```python
# Most useful sorts:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Group by input shapes (find which tensor sizes are slow):
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="cuda_time_total", row_limit=30))
```

## 2. CUDA Event Timing (Low Overhead, Always-On)

For permanent instrumentation with near-zero overhead. Uses GPU-side timestamps so it measures actual GPU time, not wall-clock.

```python
class PhaseTimer:
    """Fine-grained GPU phase timing using CUDA events."""
    
    def __init__(self):
        self.phases = {}
        self.active_phase = None
        self.start_event = None
    
    def start(self, name):
        if name not in self.phases:
            self.phases[name] = {"total_ms": 0.0, "count": 0}
        self.active_phase = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
    
    def stop(self):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()  # needed to get timing
        ms = self.start_event.elapsed_time(end)
        self.phases[self.active_phase]["total_ms"] += ms
        self.phases[self.active_phase]["count"] += 1
    
    def report(self):
        total = sum(p["total_ms"] for p in self.phases.values())
        print(f"\n{'Phase':<25} {'Total ms':>10} {'Avg ms':>10} {'Count':>6} {'%':>6}")
        print("-" * 60)
        for name, p in sorted(self.phases.items(), key=lambda x: -x[1]["total_ms"]):
            avg = p["total_ms"] / max(p["count"], 1)
            pct = p["total_ms"] / total * 100 if total > 0 else 0
            print(f"{name:<25} {p['total_ms']:>10.1f} {avg:>10.2f} {p['count']:>6} {pct:>5.1f}%")
        print(f"{'TOTAL':<25} {total:>10.1f}")
    
    def reset(self):
        self.phases.clear()

# Usage in training loop:
timer = PhaseTimer()

for iteration in range(start_iter, args.max_iters):
    timer.start("data_load")
    inp_batch, tgt_batch = next(data_iter)
    timer.stop()

    timer.start("to_device")
    inp_batch = inp_batch.to(device, non_blocking=True)
    tgt_batch = tgt_batch.to(device, non_blocking=True)
    timer.stop()

    timer.start("nafnet_forward")
    with torch.amp.autocast("cuda", enabled=use_amp):
        pred = model(inp_batch)
        pixel_loss = criterion(pred, tgt_batch)
    timer.stop()

    timer.start("vgg_forward")
    with torch.amp.autocast("cuda", enabled=False):
        p_loss = perceptual_criterion(pred.float(), tgt_batch.float())
    loss = pixel_loss + args.perceptual_weight * p_loss
    timer.stop()

    timer.start("backward")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    timer.stop()

    timer.start("optimizer_step")
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    timer.stop()

    if (iteration + 1) % 100 == 0:
        timer.report()
        timer.reset()
```

**Important caveat:** `torch.cuda.synchronize()` in `stop()` forces CPU-GPU sync each call, which itself can reduce GPU utilization. Use this for diagnostic runs only, or only call `synchronize()` + `report()` every N iterations:

```python
class AsyncPhaseTimer:
    """CUDA event timing without per-phase synchronization.
    
    Records events but only synchronizes when report() is called.
    Minimal overhead during training.
    """
    
    def __init__(self):
        self.events = []  # list of (name, start_event, end_event)
        self.current_start = None
        self.current_name = None
    
    def start(self, name):
        self.current_name = name
        self.current_start = torch.cuda.Event(enable_timing=True)
        self.current_start.record()
    
    def stop(self):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((self.current_name, self.current_start, end))
    
    def report(self):
        torch.cuda.synchronize()  # single sync to resolve all events
        totals = {}
        for name, start, end in self.events:
            ms = start.elapsed_time(end)
            if name not in totals:
                totals[name] = {"total_ms": 0.0, "count": 0}
            totals[name]["total_ms"] += ms
            totals[name]["count"] += 1
        
        total = sum(p["total_ms"] for p in totals.values())
        print(f"\n{'Phase':<25} {'Total ms':>10} {'Avg ms':>10} {'Count':>6} {'%':>6}")
        print("-" * 60)
        for name, p in sorted(totals.items(), key=lambda x: -x[1]["total_ms"]):
            avg = p["total_ms"] / max(p["count"], 1)
            pct = p["total_ms"] / total * 100 if total > 0 else 0
            print(f"{name:<25} {p['total_ms']:>10.1f} {avg:>10.2f} {p['count']:>6} {pct:>5.1f}%")
        print(f"{'TOTAL':<25} {total:>10.1f}")
        self.events.clear()
```

## 3. NVIDIA Nsight Systems on Modal

Nsight Systems gives the most detailed GPU timeline, showing individual kernel launches, memory copies, and gaps between operations.

### Can We Run It on Modal?

Yes, with caveats. Modal containers run on NVIDIA GPUs with the CUDA toolkit available. `nsys` is included in the NVIDIA container toolkit.

```python
# In your Modal image definition:
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    # nsys is pre-installed in NGC containers
    # Or on debian_slim:
    # .apt_install("nsight-systems-cli")  # may not be available
)
```

### Running nsys on Modal

```python
import subprocess

@app.function(gpu="H100", image=image, timeout=600)
def profile_training():
    # Run training with nsys profiling
    subprocess.run([
        "nsys", "profile",
        "--output", "/tmp/nafnet_profile",
        "--force-overwrite", "true",
        "--trace", "cuda,cudnn,cublas,nvtx",
        "--sample", "none",           # don't sample CPU (reduces overhead)
        "--cuda-memory-usage", "true",
        "python", "training/train_nafnet.py",
        "--max-iters", "20",
        "--data-dir", "/mnt/data/train_pairs",
    ])
    # Copy the .nsys-rep file to the volume for download
    # ...
```

### Alternative: Use NVTX Markers Without nsys

Even without nsys, NVTX markers help PyTorch's built-in profiler:

```python
import torch.cuda.nvtx as nvtx

# In training loop:
nvtx.range_push("data_loading")
inp_batch, tgt_batch = next(data_iter)
nvtx.range_pop()

nvtx.range_push("nafnet_forward")
pred = model(inp_batch)
nvtx.range_pop()

# These show up in torch.profiler traces too
```

### Getting nsys Results Back from Modal

```python
# Upload the .nsys-rep file to a Modal Volume
vol = modal.Volume.from_name("upscale-data")

@app.function(gpu="H100", volumes={"/mnt/data": vol})
def profile_and_save():
    # ... run nsys ...
    import shutil
    shutil.copy("/tmp/nafnet_profile.nsys-rep", "/mnt/data/profiles/")
    vol.commit()

# Then download locally:
# modal volume get upscale-data profiles/nafnet_profile.nsys-rep .
```

View with NVIDIA Nsight Systems GUI (free download from NVIDIA, Windows/Linux).

## 4. torch.utils.bottleneck

```bash
python -m torch.utils.bottleneck training/train_nafnet.py --max-iters 20
```

Runs the script under both cProfile and `torch.autograd.profiler`. Shows top CPU functions and top CUDA kernels. **Useful for a quick one-shot** but not for mid-training profiling. The overhead is significant (2-5x slowdown).

**Verdict:** Use `torch.profiler` instead for our use case. `bottleneck` is better for debugging startup or single-inference scripts.

## 5. What to Measure and What to Look For

### DataLoader Overlap

The critical question: **Is the next batch ready when the GPU finishes computing?**

```python
# Quick test: measure time spent in next(data_iter) vs compute
import time

data_stall_total = 0.0
compute_total = 0.0

for iteration in range(100):
    t0 = time.perf_counter()
    inp_batch, tgt_batch = next(data_iter)
    inp_batch = inp_batch.to(device, non_blocking=True)
    tgt_batch = tgt_batch.to(device, non_blocking=True)
    torch.cuda.synchronize()  # ensure transfer is done
    t1 = time.perf_counter()
    data_stall_total += t1 - t0

    # ... forward, backward, step ...
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    compute_total += t2 - t1

print(f"Data loading: {data_stall_total:.1f}s ({data_stall_total/(data_stall_total+compute_total)*100:.0f}%)")
print(f"Compute:      {compute_total:.1f}s ({compute_total/(data_stall_total+compute_total)*100:.0f}%)")
```

If data loading is >5-10% of total time, the DataLoader is the bottleneck.

**Diagnosis clues from our setup:**
- We read 1080p PNGs via cv2 and crop to 256x256. PNG decoding is CPU-intensive.
- With 8 workers and batch_size=40, each worker decodes 5 PNGs per batch.
- On Modal, CPU cores may be limited (check `cpu=` parameter in function config).

### Kernel Launch Gaps

Small models like NAFNet (14-57M params) with 256x256 crops can be **kernel launch bound** -- the CPU queues tiny kernels faster than the GPU executes them, but not fast enough to keep the GPU 100% occupied.

In a torch.profiler chrome trace, look for:
- Many short GPU kernels (<0.1ms each) with gaps between them
- CPU thread showing `cudaLaunchKernel` calls between GPU bars
- "GPU idle" percentage in the profiler summary

### VGG Forward Pass Cost

VGG19 perceptual loss runs in fp32 (to avoid numerical instability). On an H100 with batch_size=40 at 256x256:
- NAFNet forward (fp16): fast, small model
- VGG forward (fp32): **potentially 2-4x slower** because VGG19 has 143M params (vs NAFNet's 14-57M) and runs in fp32

The `record_function` annotations will reveal this split exactly.

### Memory Bandwidth vs Compute

```python
# Quick check: are we compute-bound or memory-bound?
# Small models at small resolution are almost always memory-bandwidth-bound
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Batch tensor size: {args.batch_size * 3 * args.crop_size**2 * 2 / 1024**2:.1f} MB (fp16)")

# H100 specs: 3.35 TFLOPS fp16, 3.35 TB/s HBM bandwidth
# If arithmetic intensity (FLOPs per byte loaded) is low, we're memory-bound
# NAFNet at 256x256 is almost certainly memory-bandwidth-bound
```

## 6. Minimal-Overhead Mid-Training Profiling

Pattern: profile 5-10 iterations mid-training without restarting.

```python
# Add to training loop - triggered by environment variable or iteration count
PROFILE_AT_ITER = int(os.environ.get("PROFILE_AT_ITER", -1))
PROFILE_DURATION = int(os.environ.get("PROFILE_DURATION", 5))

profiler = None
profiler_stop_iter = -1

for iteration in range(start_iter, args.max_iters):
    # Start profiler at specified iteration
    if iteration == PROFILE_AT_ITER:
        print(f"[PROFILER] Starting torch.profiler for {PROFILE_DURATION} iterations")
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # reduce overhead
        )
        profiler.__enter__()
        profiler_stop_iter = iteration + PROFILE_DURATION

    # ... training step (with record_function annotations) ...

    # Stop profiler
    if iteration + 1 == profiler_stop_iter:
        profiler.__exit__(None, None, None)
        trace_path = f"profile_iter{PROFILE_AT_ITER}.json"
        profiler.export_chrome_trace(trace_path)
        print(f"[PROFILER] Saved trace to {trace_path}")
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        profiler = None
```

**On Modal**, set the env var:

```python
@app.function(gpu="H100")
def train():
    import os
    os.environ["PROFILE_AT_ITER"] = "100"
    os.environ["PROFILE_DURATION"] = "10"
    # ... run training ...
```

## 7. Common Fixes for Low GPU Utilization

### 7.1 DataLoader Optimizations

```python
# Already doing these (good):
#   pin_memory=True
#   persistent_workers=True
#   num_workers=8

# Additional optimizations:

# 1. Prefetch factor (default=2, try higher if data loading is bottleneck)
DataLoader(..., prefetch_factor=4)

# 2. Use non_blocking transfers (we're not doing this consistently)
inp_batch = inp_batch.to(device, non_blocking=True)
tgt_batch = tgt_batch.to(device, non_blocking=True)
# non_blocking=True lets the CPU continue while GPU copies data
# IMPORTANT: only safe when the CPU doesn't read the tensor before GPU is done

# 3. Pre-decode dataset: load all PNGs into RAM at init time
#    Our dataset is ~3000 frames at 1080p = ~18GB as float32
#    On H100 with 200GB+ RAM, this eliminates all I/O
class PreloadedDataset(Dataset):
    def __init__(self, data_dir, crop_size=256):
        # Load everything into RAM at init
        self.inputs = []
        self.targets = []
        for inp_path, tgt_path in pairs:
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.inputs.append(inp)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.targets.append(tgt)
        print(f"Preloaded {len(self.inputs)} frames into RAM")

    def __getitem__(self, idx):
        pair_idx = idx % len(self.inputs)
        inp = self.inputs[pair_idx]
        tgt = self.targets[pair_idx]
        # crop + augment (no I/O, just numpy ops)
        ...

# 4. Use LMDB or numpy memmap instead of individual PNGs
#    Avoids filesystem overhead of opening thousands of files
```

### 7.2 torch.compile on Training Step

```python
# torch.compile the model (NAFNet is compile-friendly, pure CNN)
model = torch.compile(model, mode="reduce-overhead")
# "reduce-overhead" uses CUDA graphs internally

# Can also compile VGG (it's a simple sequential CNN):
if perceptual_criterion is not None:
    perceptual_criterion.vgg = torch.compile(perceptual_criterion.vgg, mode="reduce-overhead")

# Compile the full training step for maximum fusion:
@torch.compile(mode="reduce-overhead")
def train_step(model, criterion, perceptual_criterion, inp, tgt, perceptual_weight):
    with torch.amp.autocast("cuda", enabled=True):
        pred = model(inp)
        pixel_loss = criterion(pred, tgt)
    with torch.amp.autocast("cuda", enabled=False):
        p_loss = perceptual_criterion(pred.float(), tgt.float())
    loss = pixel_loss + perceptual_weight * p_loss
    return loss, pixel_loss, p_loss

# NOTE: torch.compile has warmup cost (first few iterations are slow)
# Use mode="reduce-overhead" for small models (uses CUDA graphs)
# Use mode="max-autotune" for maximum kernel fusion (longer compile)
```

### 7.3 CUDA Graphs (Manual)

CUDA graphs record a sequence of GPU operations and replay them without CPU involvement. Eliminates kernel launch overhead entirely.

```python
# CUDA graphs require static shapes (no dynamic allocation)
# Perfect for our fixed batch_size + crop_size training

# Warmup
static_inp = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size,
                          device=device, dtype=torch.float16)
static_tgt = static_inp.clone()

# Warmup run (outside graph)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        with torch.amp.autocast("cuda"):
            pred = model(static_inp)
            loss = criterion(pred, static_tgt)
        scaler.scale(loss).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
torch.cuda.current_stream().wait_stream(s)

# Capture graph
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with torch.amp.autocast("cuda"):
        static_pred = model(static_inp)
        static_loss = criterion(static_pred, static_tgt)
    scaler.scale(static_loss).backward()
    # Note: optimizer.step() inside graph is tricky with GradScaler

# Replay:
for iteration in range(...):
    static_inp.copy_(real_input)
    static_tgt.copy_(real_target)
    g.replay()
    # static_loss now has the result
```

**Caveats:** CUDA graphs don't work with:
- Dynamic shapes (our shapes are fixed, so this is fine)
- CPU-GPU sync points inside the graph
- GradScaler's dynamic loss scaling (scale changes need to be outside the graph)
- VGG perceptual loss in fp32 mixed with NAFNet in fp16 (two autocast contexts)

**Recommendation:** Use `torch.compile(mode="reduce-overhead")` instead -- it uses CUDA graphs internally but handles the edge cases.

### 7.4 Fused Optimizers

```python
# PyTorch 2.x has fused AdamW (single kernel instead of multiple per-param)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.9),
    fused=True,  # <-- fused CUDA implementation
)
# Reduces optimizer step from O(num_params) kernel launches to fewer fused kernels
# Typically 10-20% speedup on optimizer step
```

### 7.5 VGG Perceptual Loss Optimizations

The VGG loss is likely a major contributor to the 40% utilization issue:

```python
# 1. Run VGG in fp16 (if numerically stable enough)
#    Test: compare loss values in fp16 vs fp32 for 100 iterations
with torch.amp.autocast("cuda", enabled=True):  # <-- enable, not disable
    p_loss = perceptual_criterion(pred, tgt_batch)
# If loss values are similar and training is stable, keep fp16

# 2. Use fewer VGG layers (drop early layers, they're expensive and less useful)
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layers=[16, 25, 34]):  # skip layers 2, 7
        ...

# 3. Downsample before VGG (256x256 -> 128x128)
pred_down = F.interpolate(pred, scale_factor=0.5, mode="bilinear", align_corners=False)
tgt_down = F.interpolate(tgt, scale_factor=0.5, mode="bilinear", align_corners=False)
p_loss = perceptual_criterion(pred_down.float(), tgt_down.float())
# Reduces VGG compute by 4x with minimal quality impact

# 4. Compute perceptual loss every N iterations (not every iteration)
if iteration % 4 == 0:
    p_loss = perceptual_criterion(pred.float(), tgt_batch.float())
    cached_p_loss = p_loss.detach()
else:
    # Use cached value for logging, skip VGG compute
    loss = pixel_loss  # or pixel_loss + weight * cached_p_loss (no grad through VGG)
```

### 7.6 Increase Compute-to-Overhead Ratio

```python
# Larger batch size (if VRAM allows)
# H100 80GB can handle much larger batches than batch_size=40 at 256x256
# Try: 64, 96, 128

# Larger crop size (more compute per sample)
# 256 -> 384 or 512 (quadruples compute, keeps overhead constant)

# Gradient accumulation (if batch size is VRAM-limited)
accumulation_steps = 4
for micro_step in range(accumulation_steps):
    inp, tgt = next(data_iter)
    inp = inp.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        pred = model(inp)
        loss = criterion(pred, tgt) / accumulation_steps
    scaler.scale(loss).backward()
# Step once after accumulating
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

### 7.7 Reduce CPU-GPU Sync Points

```python
# Common hidden sync points that kill utilization:

# BAD: .item() forces sync
loss_val = loss.item()  # syncs!

# BETTER: log less frequently, or use async logging
if iteration % 50 == 0:
    loss_val = loss.item()  # sync only every 50 iters

# BAD: printing VRAM usage
torch.cuda.max_memory_reserved()  # syncs!

# BAD: gradient clipping with unscale
scaler.unscale_(optimizer)  # syncs internally
torch.nn.utils.clip_grad_norm_(...)  # syncs (computes norm)

# BETTER: clip less aggressively (skip if gradients are stable)
if iteration < 1000:  # only clip during early training
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

## 8. Recommended Profiling Sequence

1. **Start with CUDA event timing** (AsyncPhaseTimer above). Run 100 iterations, get the phase breakdown. This tells you immediately whether data loading, VGG forward, or backward dominates.

2. **If data loading is >10%:** Pre-load dataset into RAM, increase `prefetch_factor`, ensure Modal has enough CPU cores (`cpu=4` or higher).

3. **If VGG forward is >30%:** Try fp16 VGG, fewer layers, downsampled inputs, or compute every Nth iteration.

4. **If backward + optimizer is dominant but GPU util is still low:** Run `torch.profiler` for a chrome trace to check for kernel launch gaps. Try `torch.compile(mode="reduce-overhead")` and fused optimizer.

5. **If everything looks fast but GPU util is still ~40%:** The GPU is memory-bandwidth-bound. NAFNet at 256x256 simply doesn't have enough arithmetic intensity. Fix: larger crops, larger batch, or larger model (more middle blocks).

## 9. Complete Drop-In Profiling Snippet

Copy-paste this into `train_nafnet.py` for a one-shot diagnostic run:

```python
# Add at top of train() function, after model/optimizer setup:
ENABLE_PROFILING = os.environ.get("ENABLE_PROFILING", "0") == "1"
PROFILE_START_ITER = 10
PROFILE_ITERS = 10

if ENABLE_PROFILING:
    from torch.profiler import profile, record_function, ProfilerActivity
    timer = AsyncPhaseTimer()  # defined above
    print("[PROFILER] Will profile iterations {PROFILE_START_ITER}-{PROFILE_START_ITER + PROFILE_ITERS}")

# Then in the training loop, wrap each phase with:
#   if ENABLE_PROFILING: timer.start("phase_name")
#   ... existing code ...
#   if ENABLE_PROFILING: timer.stop()

# And add torch.profiler around the target iterations:
# if iteration == PROFILE_START_ITER:
#     profiler = profile(...)
#     profiler.__enter__()
# if iteration == PROFILE_START_ITER + PROFILE_ITERS:
#     profiler.__exit__(...)
#     profiler.export_chrome_trace(...)

# Trigger with: ENABLE_PROFILING=1 python training/train_nafnet.py --max-iters 25
```

## 10. Likely Root Causes for Our 40% GPU Utilization

Based on our setup (NAFNet 14-57M params, 256x256 crops, batch_size=40, H100):

1. **VGG perceptual loss in fp32** -- VGG19 (143M params) is larger than NAFNet and runs in fp32. This likely consumes 40-60% of the iteration time. Running it in fp16 or on downsampled inputs would help significantly.

2. **Kernel launch overhead** -- NAFNet is a small model. At 256x256, individual kernels are tiny. The CPU can't launch them fast enough to keep the H100 busy. `torch.compile` with CUDA graphs would help.

3. **DataLoader I/O** -- Reading individual PNGs from disk involves filesystem overhead. On Modal, the volume might add latency. Pre-loading into RAM eliminates this.

4. **CPU-GPU sync points** -- `loss.item()`, gradient clipping, and GradScaler all introduce sync points. Reducing logging frequency and skipping grad clip when stable would help.

5. **Arithmetic intensity too low** -- H100 has 3.35 PFLOPS of fp16 compute. At 256x256 with a 14M param model, there simply aren't enough FLOPs per iteration to keep it busy. Increasing crop_size to 512 would 4x the compute per iteration.

**Recommended first steps:**
1. Add `AsyncPhaseTimer` to get phase breakdown (1 hour to implement + run)
2. Try `torch.compile(model, mode="reduce-overhead")` (5 min change)
3. Try `fused=True` on AdamW (1 line change)
4. Try `non_blocking=True` on `.to(device)` calls (already partially done)
5. Increase crop_size to 384 or 512 if VRAM allows
