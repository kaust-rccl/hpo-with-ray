# Overview

This workshop focuses on **Hyperparameter Optimization (HPO)** for fine-tuning large language models.
You will experiment with both **manual and automated** approaches to explore how different hyperparameters affect model
**performance and training cost**.

## Key Concepts

### Hyperparameter Optimization (HPO)

- It's the process of finding the **best set of hyperparameters** (like learning rate, batch size, weight decay) that
  lead to the **best model performance**.


- Proper tuning can significantly reduce training time, GPU usage, and improve evaluation metrics.

#### Hyperparameters in This Workshop

- **Learning Rate (lr)** – Controls how much the model updates weights after each training step.

    - Too high → unstable training.

    - Too low → very slow convergence.


- **Weight Decay (wd)** – A form of regularization that prevents overfitting by penalizing large weights.


- **Batch Size (bs)** – Number of samples processed before updating model weights.

  Larger batches can speed up training but need more GPU memory.

### Evaluation Metrics

- **Evaluation Loss (eval_loss)**

    - Measures how well the model predicts on the validation dataset.

    - Lower is better (indicates better generalization).


- **GPU Hours**

    - Total amount of GPU time consumed.

    - 1 GPU hour = 1 GPU used for 1 hour (e.g., 4 GPUs × 30 minutes = 2 GPU hours).

    - Useful for comparing cost-efficiency of different methods.

### Ray Tune (Automated HPO Framework)

It's a Python library for distributed hyperparameter optimization, that **automates running multiple experiments**
in parallel and **selecting the best configurations.**
Saves time and GPU resources by **intelligently stopping poor-performing trials early** and focusing on promising ones.

### Ray Tune Schedulers

Schedulers decide how trials are run, paused, or stopped:

#### ASHA (Asynchronous Successive Halving Algorithm)

- It starts many trials with different hyperparameter combinations, then periodically **stops the worst-performing
  trials early** to free resources for better ones.

- Best used for **fast**, exploratory HPO where you want to **test many configurations quickly**.

#### PBT (Population-Based Training)

- it starts with a **"population"** of trials, then periodically **copies the weights and hyperparameters from
  top-performing trials to worse ones**.
  Also mutates (perturbs) hyperparameters dynamically **during training.**

- Best used for **long-running training** where hyperparameters might need to **change over time**.

- The “best hyperparameters” at the end are only the final phase’s values; the best weights depend on the entire
  sequence of changes.

**Quick Comparison**

- ASHA → Finds the best static hyperparameter configuration.

- PBT → Finds the best training schedule (evolving hyperparameters) to reach strong final weights.

#### Bayesian (Population-Based Training)

//TODO:

## DeepSpeed

### Purpose and Role in Training

### DeepSpeed Configuration file

---

# Environment Setup and File Structure

### Environment Setup

### Project Structure

---

# Manual HPO with SLURM:

## Overview

In this exercise, we perform **manual hyperparameter optimization (HPO)** by iterating over a predefined grid of
hyperparameters (learning rate, batch size, weight decay).
**A SLURM script** ([`manual_bloom_hpo.slurm`](./experiments/manual_hpo/manual_bloom_hpo.slurm)) handles the **job
submission, environment setup, and looping logic**, while **a Python training script** ([
`manual_bloom_hpo.py`](scripts/manual_hpo/manual_bloom_hpo.py)) handles data preprocessing, fine-tuning, and evaluation.

Each hyperparameter combination runs **sequentially** (serial execution), using 2 GPUs per run, for 5 epochs. After all
runs, we collect evaluation loss, F1, EM, and runtime to select the best configuration.

## Breaking Down the Manual HPO Building Block

### SLURM Script

This script orchestrates the training runs on the HPC cluster.

**SLURM Directives:**

- `--nodes=1`, `--gpus-per-node=2`: Single-node, 2 × V100 GPUs.

- `--cpus-per-task=6`, `--mem=256G`: Allocates CPU cores and system RAM.

- `--time=02:00:00`: Fits 12 serial runs × 5 epochs each (and some extra time just to be sure).

**Environment Setup:**

- Loads CUDA (`module load cuda/12.4.1`).

- Activates Conda environment (`conda activate hpo-raytune`).

**Hyperparameter Grid:**

- Defined as arrays:
  ```commandline
  LRs=(1e-5 2e-4 5e-6)
  BSs=(1 2)
  WDs=(0.0 0.01)
  ```
- Total combinations: 3 × 2 × 2 = 12 runs.

**Execution Loop:**

- Nested for loops iterate over **all combinations**.

- Uses torchrun `--standalone` `--nproc_per_node=2` to spawn **2 GPU processes per run**.

- Runs are executed **sequentially** (serial execution).

**Isolation Guarantee:**

By running each trial sequentially in **its own process**, we guarantee:

- **Memory isolation** – CUDA memory and CPU RAM are fully released after each trial finishes.

- **Resource isolation** – No two trials compete for the same GPUs/CPUs, preventing OOM or performance interference.

- **Checkpoint integrity** – Each run writes independently to checkpoints/, avoiding overwrite conflicts.

**Logging & Timestamps:**

- Prints job start/finish times with `trap`.

- Outputs to `logs/<job_name>-<job_id>.out`.
- Automatic Log Parsing: At the end of all trials, the SLURM script automatically generates a CSV with the same name as
  the SLURM log `logs/<job_name>-<job_id>.csv`.

### Training Python File

This script handles fine-tuning and evaluation for one hyperparameter combination.

**Argument Parser**:

- Expects `--lr`(learning rate), `--bs`(batch size), and `--wd`(weights decay) (passed from SLURM script).

**Dataset Loading & Preprocessing (load_squad())**:

- Loads SQuAD, tokenizes questions + contexts into prompts like:
  ```
  Question: <q> Context: <c> Answer:
  ```
- Pads/truncates to 512 tokens.

- Labels use -100 for padded positions (ignored during loss computation).

- Uses 1,000 training and 100 validation samples for speed.

**Model & Trainer Setup:**

- Loads BLOOM-560M.

- Configures TrainingArguments with fp16, deepspeed config, and the passed hyperparameters.

**Evaluation:**

- Runs Hugging Face trainer.evaluate() for loss.

- Custom evaluation computes:

    - **Exact Match (EM)** – strict answer match after normalization.

    - **F1** – token-level overlap score.

**Result Logging:**

- Prints a **summary dictionary** containing `lr`, `bs`, `wd`, `eval_loss`, and `runtime`.

## Exercise: Launch, Track, and Analyze

### Launching the Jobs

1. Make sure you are in the same [directory](./) as this `README`, the navigate to the SLURM script directory:
    ```commandline
    cd experiments/manual_hpo/
    ```
2. Submit the manual HPO job:
    ```commandline
    sbatch manual_bloom_hpo.slurm
    ```
3. Monitor the job in the queue
    ```commandline
    squeue --me
    ```

### Result Collection Table

- Navigate and open the logs file:
  ```commandline
  cd experiments/manual_hpo/logs
  cat bloom_hpo_serial_5_epochs-<jobid>.out
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


- Navigate and open the parsed results (done automatically after job completion):
  ```commandline
  cd experiments/manual_hpo/logs
  column -t -s, bloom_hpo_serial_5_epochs-<jobid>.csv | less -S
  ```
- Fill the result table with information extracted from the `.csv` file:

  | **Combo ID** | **Learning Rate (lr)** | **Batch Size (bs)** | **Weight Decay (wd)** | **Eval Loss** | **Runtime (s)** |
    |--------------|------------------------|----------------------|------------------------|---------------|----------------|
  | 1            | 1e-5                   | 1                    | 0.0                    |               |                |
  | 2            | 1e-5                   | 1                    | 0.01                   |               |                |
  | 3            | 1e-5                   | 2                    | 0.0                    |               |                |
  | 4            | 1e-5                   | 2                    | 0.01                   |               |                |
  | 5            | 2e-4                   | 1                    | 0.0                    |               |                |
  | 6            | 2e-4                   | 1                    | 0.01                   |               |                |
  | 7            | 2e-4                   | 2                    | 0.0                    |               |                |
  | 8            | 2e-4                   | 2                    | 0.01                   |               |                |
  | 9            | 5e-6                   | 1                    | 0.0                    |               |                |
  | 10           | 5e-6                   | 1                    | 0.01                   |               |                |
  | 11           | 5e-6                   | 2                    | 0.0                    |               |                |
  | 12           | 5e-6                   | 2                    | 0.01                   |               |                |           

### Quiz Questions

- What key information should you extract from each trial’s log to decide the best hyperparameter configuration?

---

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
  `ray_head_bloom.slurm`](./experiments/raytune_hpo/raytune_asha_scheduler/head_node_raytune_asha_hpo.slurm))
- **Worker node launcher** ([
  `worker_node_v100.slurm`](./experiments/raytune_hpo/raytune_asha_scheduler/worker_node_raytune_asha_hpo.slurm))

## Breaking Down the Ray-Tune (ASHA Scheduler) HPO Building Block

### Training Python File

### **Breaking Down the ASHA HPO Building Block**

#### **Training Python File (`bloom_ray_tune.py`)**

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
          - Small batch sizes reduce GPU memory usage (important for V100s with 32GB) but may slow convergence. Larger batch sizes speed up training but need more memory.

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

### Result Collection Table

### Quiz Questions

---

# Automated HPO with Ray Tune: Population-Based Training (PBT)

## Overview

## Breaking Down the Ray-Tune (PBT Scheduler) HPO Building Block

### Training Python File

### SLURM Script

## Exercise: Launch, Track, and Analyze

### Launching the Jobs

### Result Collection Table

### Quiz Questions

---

# Automated HPO with Ray Tune: Bayesian Optimization

## Overview

## Breaking Down the Ray-Tune (Bayesian Scheduler) HPO Building Block

### Training Python File

### SLURM Script

## Exercise: Launch, Track, and Analyze

### Launching the Jobs

### Result Collection Table

### Quiz Questions

---

# Selecting the Best Run from Each Approach