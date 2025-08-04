# Ray Tune Hyperparameter Optimization with Bayesian Optimization (BOHB)

## Overview

This experiment fine-tunes **BLOOM-560M** on the **SQuAD v1.1** dataset using **Ray Tune** with
**Bayesian Optimization and HyperBand (BOHB)**. Unlike manual tuning or ASHA, BOHB combines:

1. **Bayesian Optimization** – Learns from past trials to suggest more promising hyperparameters.  
2. **HyperBand** – Efficiently allocates resources by starting many trials, then promoting only
   the best ones for longer training.

The main training script is:

- [`raytune_bayesian.py`](./../../../../scripts/raytune/scheduler/bayesian/raytune_bayesian_hpo.py)

The cluster orchestration uses SLURM scripts:

- [`head_node_raytune_bayesian_hpo.slurm`](./head_node_raytune_bayesian_hpo.slurm)
- [`worker_node_raytune_bayesian_hpo.slurm`](./worker_node_raytune_bayesian_hpo.slurm)

> Users only need to submit the **head node job**. The head script automatically spawns the worker jobs.

---

## Training Logic

### Python Script (`raytune_bayesian.py`)

- **Dataset**: Loads SQuAD v1.1 (~85k training / 10k validation examples).  
- **Model**: BLOOM-560M via Hugging Face Transformers.  
- **Training Backend**: Hugging Face Trainer + DeepSpeed + fp16 mixed precision.  
- **Evaluation**: Uses both Hugging Face's `eval_loss` and custom metrics:
  - **Exact Match (EM)**
  - **F1 Score**

**Hyperparameters Tuned**:

```python
"train_loop_config": {
    "lr": tune.loguniform(5e-6, 2e-4),
    "per_device_bs": tune.choice([1, 2]),
    "wd": tune.choice([0.0, 0.01]),
}
```

**Scheduler Configuration**:

```python
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=50,
    reduction_factor=4,
    stop_last_trials=False,
)
```

---

## SLURM Job Scripts

### Head Node (`head_node.slurm`)
- Starts the Ray head node with dynamic ports.  
- Spawns worker jobs automatically.  
- Launches the main Python script (`raytune_bayesian.py`).  
- Shuts down Ray after completion.

### Worker Node (`worker_node.slurm`)
- Connects to the Ray head node.  
- Uses allocated GPUs and CPUs to run Ray workers.  
- Waits for a shutdown signal.

> You **only submit the head node job**; it manages the workers.

---

## How Bayesian Optimization Works

Imagine a **chef experimenting with recipes**:

- **ASHA**: Throw many dishes together randomly and quickly discard the bad ones.  
- **Bayesian Optimization**: Taste some dishes, **learn what works**, and refine recipes in later attempts.  
- **BOHB**: Combines both – start with many small tastings, keep only the best, and refine them further.

This makes BOHB **smarter than ASHA** when compute budget is limited.

---

## Running the Experiment

1. Navigate to the experiment folder:

```bash
cd experiments/raytune_hpo/raytune_bayesian/
```

2. Submit the job using `sbatch`, and optionally override the search space hyperparameters using environment variables:

    ```commandline
    LR_LOWER=1e-5 \
    LR_UPPER=2e-4 \
    BS_CHOICES="1 2" \
    WD_CHOICES="0.0 0.01" \
    sbatch head_node_raytune_asha_hpo.slurm
    ```
   You can customize the following variables:

    - `LR_LOWER`: Lower bound of learning rate range
    - `LR_UPPER`: Upper bound of learning rate range
    - `BS_CHOICES`: Space-separated list of per-device batch sizes
    - `WD_CHOICES`: Space-separated list of weight decay values
   

3. Monitor the job queue:

```bash
squeue --me
```

---

## Viewing Results

Logs will be saved in the `logs/` directory. After completion, you can view results with:

```bash
cd logs/
cat ray_head_bloom_bohb-<jobid>.out
```

At the end, Ray Tune prints the **Best Trial Result**, e.g.:

```json
Best Trial Result:
{
  "eval_loss": 9.24,
  "exact_match": 0.71,
  "f1": 0.76,
  "config": {
    "train_loop_config": {
      "lr": 2.5e-05,
      "per_device_bs": 2,
      "wd": 0.0
    }
  }
}
```

### Example Results Table

  | **Combo ID** | **Learning Rate (lr)** | **Batch Size (bs)** | **Weight Decay (wd)** | **Eval Loss** | **Runtime (s)** |
  |--------------|------------------------|---------------------|-----------------------|---------------|-----------------|
  | 1            |                        |                     |                       |               |                 |
  | 2            |                        |                     |                       |               |                 |
  | 3            |                        |                     |                       |               |                 |
  | 4            |                        |                     |                       |               |                 |
  | 5            |                        |                     |                       |               |                 |
  | 6            |                        |                     |                       |               |                 |
  | 7            |                        |                     |                       |               |                 |
  | 8            |                        |                     |                       |               |                 |
  | 9            |                        |                     |                       |               |                 |
  | 10           |                        |                     |                       |               |                 |
  | 11           |                        |                     |                       |               |                 |
  | 12           |                        |                     |                       |               |                 |

---

## Quiz Questions

1. How does BOHB balance **exploration** vs **exploitation** when compared to ASHA?  
2. Why might some GPUs appear idle during later stages of a BOHB run?  
3. If you want to keep GPUs busier, how could you adjust `reduction_factor` or `max_concurrent_trials`?  

---
