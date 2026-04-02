# Modal Graceful Shutdown Research

Research into gracefully stopping a running Modal training function and saving state before termination.

## Summary of Findings

Modal does support graceful shutdown, but the mechanisms require restructuring our training code. The best approach combines a **Volume-based stop flag** with **`@modal.exit()` lifecycle hooks** on a `modal.Cls`. Modal's recommended pattern for long training is **not** graceful mid-run stopping, but rather checkpoint-and-retry with short timeouts.

---

## 1. SIGTERM / Signal Handling

**Modal sends SIGINT (not SIGTERM) on preemption and container stop.** This raises `KeyboardInterrupt` in Python.

From the docs on `modal.exception.simulate_preemption`:
- First interrupt: **SIGINT** signal (catchable as `KeyboardInterrupt`)
- After **30 seconds**: a second interrupt simulates **SIGKILL** (uncatchable, container dies)

So there is a **30-second grace period** between SIGINT and forced kill.

**`modal container stop <container_id>`** explicitly sends SIGINT to the container. This is the per-container equivalent of stopping.

**`modal app stop <app_name>`** stops the entire app. The docs don't specify a grace period for `app stop` vs `container stop`, but the preemption docs indicate the same SIGINT -> 30s -> SIGKILL pattern applies universally.

### How to use this

Wrap the training loop in a `try/except KeyboardInterrupt`:

```python
try:
    for iteration in range(start_iter, max_iters):
        # ... training step ...
except KeyboardInterrupt:
    print("Caught interrupt, saving checkpoint...")
    save_checkpoint(model, optimizer, scheduler, scaler, iteration, ckpt_dir)
    vol.commit()
    print("Checkpoint saved, exiting.")
    raise  # Re-raise so Modal sees the interruption
```

**Caveat:** You have only 30 seconds to save. A NAFNet checkpoint is ~1.3GB (model + optimizer), and `vol.commit()` needs to flush to the distributed filesystem. This should be feasible but tight for large checkpoints.

---

## 2. `@modal.exit()` Lifecycle Hooks

**Yes, Modal supports exit hooks via `@modal.exit()` on a `modal.Cls`.** This requires converting from `@app.function` to `@app.cls`.

From the docs:
> The container exit handler is called when a container is about to exit. It is useful for doing one-time cleanup, such as closing a database connection or saving intermediate results.
>
> Exit handlers are also called when a container is preempted. The exit handler is given a **grace period of 30 seconds** to finish, and it will be killed if it takes longer than that to complete.

### Implementation pattern

```python
@app.cls(gpu="H100", volumes={VOL_MOUNT: vol}, timeout=28800)
class Trainer:
    @modal.enter()
    def setup(self):
        self.model = None
        self.optimizer = None
        self.iteration = 0
        self.should_stop = False

    @modal.method()
    def train(self, data_dir, checkpoint_dir, ...):
        # ... setup model, optimizer, dataloader ...
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir

        for iteration in range(start_iter, max_iters):
            # ... training step ...
            self.iteration = iteration

    @modal.exit()
    def save_on_exit(self):
        if self.model is not None:
            print(f"Exit hook: saving checkpoint at iteration {self.iteration}...")
            save_checkpoint(self.model, self.optimizer, ...)
            vol.commit()
            print("Checkpoint saved.")
```

**Key detail:** The exit hook runs in the same container, so it has access to `self.model` etc. The 30-second grace period is the same as for signal handling. Using `@modal.exit()` is cleaner than `try/except KeyboardInterrupt` because it's guaranteed to run on any container shutdown (preemption, timeout, app stop).

**Limitation:** `@modal.exit()` still only gets 30 seconds. If saving + committing takes longer, the container is killed.

---

## 3. Signaling via Modal Dict or Queue

**Yes, `modal.Dict` can be used to send a signal to a running function.** This is the most controllable approach.

Modal Dicts are persistent distributed key-value stores, accessible from anywhere:

```python
# Create a shared signal dict
signal = modal.Dict.from_name("train-signal", create_if_missing=True)

# In the training loop (inside the remote function):
for iteration in range(start_iter, max_iters):
    # Check for stop signal every N iterations
    if iteration % 100 == 0:
        try:
            if signal.get("stop", default=False):
                print(f"Stop signal received at iteration {iteration}")
                save_checkpoint(...)
                vol.commit()
                signal["stop"] = False  # Reset the flag
                return
        except Exception:
            pass  # Don't crash training if Dict is unavailable
    # ... training step ...
```

To trigger the stop from your local machine:
```python
import modal
signal = modal.Dict.from_name("train-signal")
signal["stop"] = True
print("Stop signal sent. Training will stop after current iteration.")
```

Or from the CLI, create a tiny script:
```bash
# stop_training.py
import modal
signal = modal.Dict.from_name("train-signal")
signal["stop"] = True
print("Stop signal sent.")
```

**Pros:**
- Clean, controlled shutdown at a natural boundary (end of iteration)
- No 30-second time pressure -- the training loop saves at its own pace
- Can be triggered from any machine with Modal access
- Can add other signals (e.g., "save now but keep training", "reduce LR")

**Cons:**
- Adds a network round-trip every N iterations (negligible if checking every 100 iters)
- Requires Modal Dict to be accessible (rare failure mode)
- Training code needs modification

`modal.Queue` could also work but Dict is simpler for a boolean flag.

---

## 4. Volume File as a Stop Flag

**Yes, this works and is even simpler than Modal Dict.** Modal Volumes support background commits, so a file written from outside will eventually appear.

However, there is a critical detail: **Volume reads within a running container see a cached/stale view unless you call `vol.reload()`.** From the docs:
> Background commits: every few seconds while your Function executes, the contents of attached Volumes will be committed without your application code calling `.commit`.

But for *reading* external changes, you need `vol.reload()`:

```python
# In training loop:
if iteration % 500 == 0:
    vol.reload()  # Fetch latest volume state
    if os.path.exists("/mnt/data/STOP"):
        print("Stop flag found, saving and exiting...")
        save_checkpoint(...)
        vol.commit()
        os.remove("/mnt/data/STOP")
        vol.commit()
        return
```

To send the stop signal:
```python
import modal
vol = modal.Volume.from_name("upscale-data")
with vol.batch_upload() as batch:
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop")
        tmp = f.name
    batch.put_file(tmp, "/STOP")
    os.unlink(tmp)
```

**Pros:**
- No additional Modal resources (Dict) needed
- Uses the Volume you already have

**Cons:**
- `vol.reload()` has overhead (more than a Dict lookup)
- Latency: Volume consistency is eventually-consistent, so there may be a delay before the file appears
- More moving parts than Dict (file creation, reload timing)

**Verdict:** Modal Dict is better for signaling. Use Volume for checkpoint data, Dict for control signals.

---

## 5. Container Lifecycle on `modal app stop`

**What happens when you run `modal app stop`:**

1. Modal sends **SIGINT** to all running containers in the app
2. This triggers `KeyboardInterrupt` in Python and/or the `@modal.exit()` handler
3. Containers get a **30-second grace period**
4. After 30 seconds, containers are forcefully killed (SIGKILL)

**What happens on preemption** (same behavior):
> Modal will send an interrupt signal to your container when preemption occurs. This will cause the Function's exit handler to run, which can perform any cleanup within its grace period.

**Note:** GPU Functions cannot be made non-preemptible (`nonpreemptible` parameter is not supported for GPU Functions). So preemption tolerance is essential for any long GPU training job.

---

## 6. Modal's Recommended Pattern: Short Timeouts + Retries

Modal's official long-training example uses a fundamentally different approach: **don't try to gracefully stop -- instead, make training naturally interruptible.**

From `modal.com/docs/examples/long-training`:

```python
@app.function(
    volumes=volumes,
    gpu="a10g",
    timeout=30,  # Short timeout!
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    single_use_containers=True,
)
def train_interruptible(*args, **kwargs):
    train(*args, **kwargs)
```

The pattern:
1. Set a short timeout (e.g., 1 hour instead of 8 hours)
2. Save checkpoints frequently (every N iterations)
3. On startup, check for and resume from the latest checkpoint
4. Use `modal.Retries` to automatically restart after timeout/preemption
5. Use `.spawn(...).get()` instead of `.remote()` for calls exceeding 24 hours

**Pros:**
- No custom signaling needed
- Naturally handles preemption, timeouts, and crashes
- Maximum progress loss = time since last checkpoint

**Cons:**
- Container restarts have overhead (image load, model init, data loading)
- Not responsive to user "stop now" commands

---

## Recommended Approach for Our Training

Combine multiple strategies for maximum robustness:

### 1. Convert to `modal.Cls` with `@modal.exit()` (safety net)

Ensures checkpoint is saved on any container shutdown (preemption, timeout, app stop). This is the baseline safety net.

### 2. Add Modal Dict stop signal (user-initiated graceful stop)

Check a `modal.Dict` flag every N iterations. When set, finish the current iteration, save checkpoint, commit volume, and exit cleanly. This gives you full control.

### 3. Keep frequent checkpointing (crash tolerance)

Continue saving at `save_freq` intervals. With background Volume commits, these are automatically persisted.

### 4. Add `KeyboardInterrupt` handler (belt and suspenders)

Catch SIGINT in the training loop as a fallback, in case the Dict check hasn't run recently.

### Implementation sketch

```python
import modal

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
signal_dict = modal.Dict.from_name("train-signal", create_if_missing=True)

@app.cls(gpu="H100", volumes={VOL_MOUNT: vol}, timeout=28800)
class Trainer:
    @modal.enter()
    def setup(self):
        self.model = None
        self.state = None

    @modal.method()
    def train(self, **kwargs):
        # ... model/optimizer setup ...
        self.model = model
        self.state = {"optimizer": optimizer, "scheduler": scheduler, ...}
        self.ckpt_dir = checkpoint_dir

        try:
            for iteration in range(start_iter, max_iters):
                # Check stop signal every 100 iterations
                if iteration % 100 == 0:
                    try:
                        if signal_dict.get("stop", default=False):
                            print(f"Stop signal at iter {iteration}")
                            self._save(iteration, "signal")
                            signal_dict["stop"] = False
                            return
                    except Exception:
                        pass

                # ... training step ...

                # Regular checkpoint
                if (iteration + 1) % save_freq == 0:
                    self._save(iteration + 1, "periodic")

        except KeyboardInterrupt:
            print(f"Interrupted at iter {iteration}")
            self._save(iteration, "interrupt")
            raise

    def _save(self, iteration, reason):
        print(f"Saving checkpoint ({reason}) at iteration {iteration}...")
        torch.save({...}, os.path.join(self.ckpt_dir, "nafnet_latest.pth"))
        vol.commit()

    @modal.exit()
    def on_exit(self):
        if self.model is not None:
            print("Exit hook: saving emergency checkpoint...")
            self._save(getattr(self, '_current_iter', 0), "exit_hook")
```

### Sending the stop signal

Create a small utility script `tools/stop_training.py`:

```python
import modal

signal_dict = modal.Dict.from_name("train-signal")
signal_dict["stop"] = True
print("Stop signal sent. Training will finish current iteration and save.")
```

Run with: `modal run tools/stop_training.py` or just `python tools/stop_training.py`

---

## Key Facts Summary

| Mechanism | Grace Period | Trigger | Requires Code Change |
|---|---|---|---|
| `modal app stop` | 30s (SIGINT -> SIGKILL) | CLI/UI | No (but loses unsaved work) |
| `modal container stop` | 30s (SIGINT) | CLI | No (same as above) |
| `@modal.exit()` hook | 30s | Any container shutdown | Yes (convert to Cls) |
| `KeyboardInterrupt` catch | 30s | SIGINT from preemption/stop | Yes (try/except) |
| Modal Dict flag | Unlimited | User sets flag | Yes (check in loop) |
| Volume stop file | Unlimited (but stale reads) | User writes file | Yes (reload + check) |
| Short timeout + retries | N/A (auto-restart) | Timeout expiry | Yes (restructure) |

**Best combination for our use case:** Modal Dict flag (primary) + `@modal.exit()` hook (safety net) + `KeyboardInterrupt` catch (belt and suspenders).
