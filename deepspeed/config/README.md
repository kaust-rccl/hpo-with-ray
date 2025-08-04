### DeepSpeed Configuration File

This experiment uses a DeepSpeed configuration that enables memory-efficient training, automatic scaling, and mixed precision. Below is a breakdown of its key components:

#### General Training Settings

- **`train_micro_batch_size_per_gpu: "auto"`**  
  Automatically selects the largest micro-batch size that fits in the available GPU memory.

- **`gradient_accumulation_steps: "auto"`**  
  Automatically adjusts accumulation steps to maintain a constant global batch size across different GPU setups.

#### Optimizer

- **Type**: `AdamW`  
- **Parameters**:
  - `lr`: `"auto"` — dynamically set by Ray Tune.
  - `betas`: `[0.9, 0.999]` — standard momentum parameters.
  - `eps`: `1e-8` — for numerical stability.

#### ZeRO Optimization (Stage 3)

- **`stage: 3`**  
  Partitions model **parameters**, **gradients**, and **optimizer states** across all GPUs to drastically reduce memory usage.

- **Offloading**:  
  - `offload_param.device`: `"none"` — all parameters stay on GPU.  
  - `offload_optimizer.device`: `"none"` — optimizer state remains on GPU for speed.

- **Communication Optimizations**:
  - `allgather_partitions`: `true` — partitions are gathered only when needed.
  - `allgather_bucket_size`: `2e8` (200MB)
  - `reduce_scatter`: `true` — enables more efficient gradient communication.
  - `reduce_bucket_size`: `2e8` (200MB)
  - `overlap_comm`: `true` — overlaps communication with computation to reduce idle time.

#### FP16 Mixed Precision

- `enabled`: `true` — activates 16-bit precision training.
- `loss_scale`: `512` — initial static loss scaling value.
- `loss_scale_window`: `1000` — window size for dynamic adjustment.
- `hysteresis`: `2` — controls scaling adjustment tolerance.
- `min_loss_scale`: `64` — prevents loss scale from shrinking too much.

---

This configuration enables **efficient, stable, and scalable training** of large models like BLOOM-560M, even on limited GPU resources.
