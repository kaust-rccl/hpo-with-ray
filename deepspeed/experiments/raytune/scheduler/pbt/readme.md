
# Automated HPO with Ray Tune: Population-Based Training (PBT)

## Overview

In this exercise, we perform **automated hyperparameter optimization** using **Ray Tune’s Population-Based Training (PBT)** scheduler.  
Unlike ASHA, which stops bad trials early, **PBT periodically explores new hyperparameter combinations** during training by **cloning and mutating** well-performing trials into weaker ones.

The training is executed via a **Python script** ([`raytune_pbt_hpo.py`](./../../../../scripts/raytune/scheduler/pbt/raytune_pbt_hpo.py))  
and orchestrated on the cluster using:

- **Head node SLURM script** ([`head_node_raytune_pbt_hpo.slurm`](./head_node_raytune_pbt_hpo.slurm))
- **Worker node SLURM script** ([`worker_node_raytune_pbt_hpo.slurm`](./worker_node_raytune_pbt_hpo.slurm))

## Breaking Down the Ray-Tune (PBT Scheduler) HPO Building Block

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

- Key PBT Configuration:

    ```python
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_loss",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations={
            "train_loop_config": {
                "lr": tune.loguniform(5e-6, 2e-4),
                "per_device_bs": [1, 2],
                "wd": [0.0, 0.01]}
        },
    )
    ```

  **Explanation of Parameters:**

  - **`time_attr="training_iteration"`**  
    Each **iteration** is a full cycle of `save_steps`, `eval_steps`, and `checkpoint/report`.  
    It aligns with one outer training cycle, not a single batch or step.

  - **`perturbation_interval=1`**  
    Every 1 iteration, PBT attempts to **exploit (clone)** the best trials into weaker ones and **explore (mutate)** the hyperparameters.

  - **`hyperparam_mutations`**  
    Defines how new values are generated. You can:
    - List categorical options (`[1, 2]`)
    - Use `lambda` or `tune.loguniform` for continuous sampling.


### SLURM Scripts

The SLURM setup for PBT follows the same principles as the manual run — environment preparation, CUDA/Conda setup, and
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
    - PBT isolates resources **per worker**, allowing multiple trials to share the cluster simultaneously.


- **Automatic Shutdown:**
    - After all trials finish, the Ray head signals the workers to stop by creating `shutdown.txt`.

### **Quick-Reference: Calculating Worker Node Resources**

When running PBT (or any Ray Tune scheduler), it’s important to size worker nodes properly.  
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
    sbatch head_node_raytune_pbt_hpo.slurm
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
  cd ./logs
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

- Scroll inside the log to locate the Ray Tune trials table (PBT prints it automatically).
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

> You run 8 trials in Ray Tune using PBT. Each trial takes ~20 minutes, but the full job finishes in ~25 minutes. Why is the total job time much **less** than the sum of individual runtimes?

- [ ] A. Ray skips some of the trials  
- [ ] B. The cluster was underutilized  
- [ ] C. Trials ran in parallel and bad ones were stopped early  
- [ ] D. Ray compresses time by batching trial steps

#### 2. Advantages of PBT vs Manual Tuning

> Compared to manual HPO, what are advantages of using PBT for hyperparameter optimization?

- [ ] A. It reduces total runtime by running trials concurrently and stopping weak ones early  
- [ ] B. It allows exploring more hyperparameter combinations in the same time window  
- [ ] C. It guarantees to find the global best configuration  
- [ ] D. It trains every trial for the full number of epochs

(*Select all that apply*)

---
