# Automated HPO with Ray Tune: ASHA Scheduler

## Overview

In this exercise, we perform **automated hyperparameter optimization** using **Ray Tune’s ASHA (Asynchronous Successive
Halving Algorithm)** scheduler.  
Unlike the manual grid search, ASHA runs **multiple trials concurrently** and **stops poor-performing trials early**,
freeing resources for more promising ones.

The training is handled by a **Python script** ([
`bloom_ray_tune.py`](./scripts/raytune_hpo/raytune_asha_scheduler/raytune_asha_hpo.py)),  
while job orchestration on the HPC cluster is handled by two **SLURM scripts**:

- **Head node launcher** ([
  `ray_head_bloom.slurm`](experiments/raytune/scheduler/asha/head_node_raytune_asha_hpo.slurm))
- **Worker node launcher** ([
  `worker_node_v100.slurm`](experiments/raytune/scheduler/asha/worker_node_raytune_asha_hpo.slurm))

## Breaking Down the Ray-Tune (ASHA Scheduler) HPO Building Block

### Training Python File

- **Dataset Loading & Preprocessing:**
    - Uses the same `load_squad()` preprocessing as in manual HPO.
    - Trains on a subset: **1,000 training samples, 100 validation samples**.


- **Model Setup:**
    - Loads **BLOOM-560M**.
    - Uses `TrainingArguments` with **deepspeed config**, mixed precision (fp16), and 5 epochs.


- **Ray Tune Integration:**
    - Reports evaluation loss (`eval_loss`) back to Ray after each evaluation step using a custom `TuneReportCallback`.
    - `train_loop_per_worker()` wraps the Hugging Face `Trainer` and is executed per Ray worker.


- **Hyperparameter Search Space:**
    - In the manual script, hyperparameters are explicitly passed via command-line args (`--lr`, `--bs`, `--wd`).
    - Here, they come from Ray Tune’s `train_loop_config`, which samples combinations automatically:
    ```python
    "train_loop_config": {
        "lr": tune.loguniform(5e-6, 2e-4),
        "per_device_bs": tune.choice([1, 2]),
        "wd": tune.choice([0.0, 0.01])
    }
    ```
  **Explanation of Each Argument:**

  1. **`lr` (Learning Rate)**  
   - **`tune.loguniform(5e-6, 2e-4)`**  
   - Ray Tune will randomly sample learning rates on a **logarithmic scale** between **5×10⁻⁶** and **2×10⁻⁴**.  
   - The logarithmic scale is preferred because learning rates can vary over orders of magnitude, and small changes at lower values often have a big effect on training stability.

   2. **`per_device_bs` (Per-Device Batch Size)**  
      - **`tune.choice([1, 2])`**  
      - Ray Tune will pick either **1** or **2** samples per GPU per forward/backward pass.  
      - Small batch sizes reduce GPU memory usage (important for V100s with 32GB) but may slow convergence. 
      
        Larger batch sizes speed up training but need more memory.

   3. **`wd` (Weight Decay)**  
      - **`tune.choice([0.0, 0.01])`**  
      - Ray Tune will pick either **no weight decay (0.0)** or a small regularization term (**0.01**).  
      - Weight decay helps reduce overfitting by penalizing large weights, but too much can slow learning.

  **How ASHA Uses These Values:**
    - ASHA will start multiple trials with different sampled combinations.
    - Poor-performing combinations (high `eval_loss`) will be stopped early, while promising ones continue training
      longer.


- **ASHA Scheduler:**
    - Stops low-performing trials early.
    - Configured with:
  ```python
  scheduler = tune.schedulers.ASHAScheduler(
    metric="eval_loss",
    mode="min",
    grace_period=1,
    max_t=5,
    reduction_factor=2
  )
  ```

  **Explanation of Each Argument:**

    1. **`metric="eval_loss"`**
        - The scheduler monitors the **evaluation loss** reported by each trial.
        - **Lower** values indicate better performance.

    2. **`mode="min"`**
        - Tells Ray Tune to **minimize** the specified metric (`eval_loss`).

    3. **`grace_period=1`**
        - The **minimum number of training iterations (epochs)** each trial will run **before ASHA considers stopping it
          **.
        - A grace period **prevents stopping trials too early** due to random noise in the first few steps.

    4. **`max_t=5`**
        - The **maximum number of training iterations (epochs)** allowed for any trial.
        - Since our fine-tuning is capped at 5 epochs, this ensures no trial runs beyond that.

    5. **`reduction_factor=2`**
        - Determines how aggressively ASHA prunes bad trials.
        - After each "bracket" (promotion round), only the **top 1/reduction_factor (e.g., 50%)** of trials continue to
          the next stage.
        - Larger values = more aggressive early stopping (faster, cheaper, but riskier).

  **How It Works in Practice:**
    - All trials start training and are periodically compared based on `eval_loss`.
    - After the grace period, poorly performing trials are stopped, and resources are reallocated to better ones.
    - Over time, only the most promising configurations continue training to the full 5 epochs.

- **Code Structure:**
    - The training logic is wrapped in `train_loop_per_worker()` for Ray, instead of a single `main()` function.
    - At the end, `tuner.fit()` automatically handles running, tracking, and storing all trials — no need for manual
      looping or parsing.


- **Reporting Metrics:**
    - In the manual script, evaluation metrics (`eval_loss`, runtime) are just printed at the end of training.
    - Here, metrics are reported **after each evaluation step** using a `TuneReportCallback`, allowing ASHA to make
      early-stopping decisions.

### SLURM Scripts

The SLURM setup for ASHA follows the same principles as the manual run — environment preparation, CUDA/Conda setup, and
logging — but with key adjustments to enable **distributed and concurrent trials**:

- **Head Node Script:**
    - Similar to the manual script, it logs **job start and finish times** with `trap`.
    - Unlike the manual script, it:
        - Starts a **Ray head node** and allocates **dynamic ports** for dashboard and worker communication.
        - Spawns **worker jobs** (`worker_node_v100.slurm`) automatically instead of looping through trials within one
          job.
        - Does not manually call the Python script per trial — instead, it runs `bloom_ray_tune.py` once, and Ray
          handles all trial scheduling internally.


- **Worker Node Script:**
    - Not used in the manual run — in manual HPO, only a single SLURM job is needed since trials run sequentially on one
      node.
    - Here, workers are **separate SLURM jobs** that join the Ray head, enabling **concurrent trials across multiple
      nodes**.
    - Allocates full node resources (e.g., 8 × V100 GPUs per worker) instead of the **2 GPUs per trial** setup in the
      manual run.


- **Resource Allocation Philosophy:**
    - Manual HPO isolates resources **per trial** by running one after another.
    - ASHA isolates resources **per worker**, allowing multiple trials to share the cluster simultaneously.


- **Automatic Shutdown:**
    - After all trials finish, the Ray head signals the workers to stop by creating `shutdown.txt`.

### **Quick-Reference: Calculating Worker Node Resources**

When running ASHA (or any Ray Tune scheduler), it’s important to size worker nodes properly.  
Each worker (Ray Tune → `ScalingConfig.num_workers`) runs **one trial at a time**, but multiple trials can land on the
same worker node concurrently if resources allow.

#### **Symbols**

| Symbol    | Meaning                                                                        |
|-----------|--------------------------------------------------------------------------------|
| **W**     | Number of workers (`ScalingConfig.num_workers`)                                |
| **Cw**    | CPUs requested per worker (`resources_per_worker["CPU"]`)                      |
| **Gw**    | GPUs requested per worker (`resources_per_worker["GPU"]`)                      |
| **Csys**  | System overhead per node (~2 CPU for raylet + object store)                    |
| **Tnode** | Maximum trials that might run **simultaneously** on one worker node            |
| **Cnode** | Total CPUs physically available on the node (`--num-cpus` / `--cpus-per-task`) |
| **Gnode** | Total GPUs physically available on the node (`--gpus-per-node`)                |

#### **Capacity Checks**

A worker node can run `Tnode` concurrent trials only if:

1. **CPU Check**
    ```
    Csys + Tnode * W * Cw <= Cnode
    ```

2. **GPU Check**
    ```
    Tnode * W * Gw <= Gnode
    ```

If either inequality fails, the extra trial will remain **PENDING** until resources free up.

#### **Example: Current Configuration**

- **W = 2**, **Cw = 2**, **Gw = 1**
- **Csys ≈ 1**, **Cnode = 20**, **Gnode = 8**
- Want: **Tnode = 4** trials concurrently on one node.

**CPU Check:**

```
2 + (4 * 2 * 2) = 18 <= 20
```

**GPU Check:**

```
4 * 2 * 1 = 8 <= 8
```

✔ **Result:** This worker node can support up to 4 concurrent trials.

#### **How to Use This**

- **Before submitting jobs:**
    1. Estimate **Tnode** (how many trials you want per node).
    2. Adjust **W**, **Cw**, **Gw** in the Ray `ScalingConfig` until both checks pass.

- **If you see PENDING trials:**
    - Check these inequalities first — insufficient CPUs or GPUs on worker nodes is the most common cause.

## Exercise: Launch, Track, and Analyze

### Launching the Jobs

1. Make sure you are in the same [directory](./) as this `README`.

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
 

3. Monitor the job in the queue:
    ```commandline
    squeue --me
    ```

### Result Collection Table

- Navigate and open the Ray Tune logs file (produced by the head SLURM script):
  ```commandline
  cd experiments/raytune_hpo/raytune_asha_scheduler/logs
  cat ray_head_bloom_5epochs-<jobid>.out
  ```
  - Find the logged job start and finish time, it should look like:
      ```commandline
      ===== JOB 39567495 START  : yyyy-mm-dd hh:mm:ss +03 =====
      ...
      ===== JOB 39567495 FINISH : yyyy-mm-dd hh:mm:ss +03 =====
    
      ```
    | **Job Start Time** | **Job Finish Time** | **Total Job Time ** |
    |--------------------|---------------------|---------------------|
    | <br/>              |                     |                     |

- Scroll inside the log to locate the Ray Tune trials table (ASHA prints it automatically).
  It will look similar to:
    ```commandline
    ╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ Trial name              status         train_loop_config/lr     ...fig/per_device_bs     train_loop_config/wd     iter     total time (s)     eval_loss │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ TorchTrainer_c89517f2   TERMINATED              5.39429e-05                        2                     0           5            484.505      10.2314  │
    │ TorchTrainer_46b3bb6c   TERMINATED              5.64985e-06                        1                     0.01        5            793.556       9.27868 │
    │ ...                                                                                                                                                      │
    ╰─────────────────
    ```
- Extract trial details to fill the following table:

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

- At the bottom of the log, find the Best Trial Result printed by Ray Tune, it should be similar to:

```commandline
Best Trial Result:
{'eval_loss': 9.24211, 'epoch': 2.0, 'time_total_s': 201.80,
 'config': {'train_loop_config': {'lr': 2.53307e-05, 'per_device_bs': 2, 'wd': 0.0}}}
```

| **Best Learning Rate (lr)** | **Best Batch Size (bs)** | **Best Weight Decay (wd)** | **Best Eval Loss** | **Total Runtime (s)** | **Epochs** |
|-----------------------------|--------------------------|----------------------------|--------------------|-----------------------|------------|
| <br/>                       |                          |                            |                    |                       |            |

### Quiz Questions

#### 1. Total Job Time vs. Trial Runtimes

> You run 8 trials in Ray Tune using ASHA. Each trial takes ~20 minutes, but the full job finishes in ~25 minutes. Why is the total job time much **less** than the sum of individual runtimes?

- [ ] A. Ray skips some of the trials  
- [ ] B. The cluster was underutilized  
- [ ] C. Trials ran in parallel and bad ones were stopped early  
- [ ] D. Ray compresses time by batching trial steps

#### 2. Advantages of ASHA vs Manual Tuning

> Compared to manual HPO, what are advantages of using ASHA for hyperparameter optimization?

- [ ] A. It reduces total runtime by running trials concurrently and stopping weak ones early  
- [ ] B. It allows exploring more hyperparameter combinations in the same time window  
- [ ] C. It guarantees to find the global best configuration  
- [ ] D. It trains every trial for the full number of epochs

(*Select all that apply*)

---
