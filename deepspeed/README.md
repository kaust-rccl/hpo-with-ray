# Overview

This workshop focuses on **Hyperparameter Optimization (HPO)** for fine-tuning large language models.
You will experiment with both **manual and automated** approaches to explore how different hyperparameters affect model **performance and training cost**.

## Key Concepts 

### Hyperparameter Optimization (HPO)

- It's the process of finding the **best set of hyperparameters** (like learning rate, batch size, weight decay) that lead to the **best model performance**.


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

- It starts many trials with different hyperparameter combinations, then periodically **stops the worst-performing trials early** to free resources for better ones.

- Best used for **fast**, exploratory HPO where you want to **test many configurations quickly**.

#### PBT (Population-Based Training)

- it starts with a **"population"** of trials, then periodically **copies the weights and hyperparameters from top-performing trials to worse ones**.
Also mutates (perturbs) hyperparameters dynamically **during training.**

- Best used for **long-running training** where hyperparameters might need to **change over time**.

- The “best hyperparameters” at the end are only the final phase’s values; the best weights depend on the entire sequence of changes.


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

In this exercise, we perform **manual hyperparameter optimization (HPO)** by iterating over a predefined grid of hyperparameters (learning rate, batch size, weight decay).
**A SLURM script** ([`manual_bloom_hpo.slurm`](./experiments/manual_hpo/manual_bloom_hpo.slurm)) handles the **job submission, environment setup, and looping logic**, while **a Python training script** ([`manual_bloom_hpo.py`](scripts/manual_hpo/manual_bloom_hpo.py)) handles data preprocessing, fine-tuning, and evaluation.

Each hyperparameter combination runs **sequentially** (serial execution), using 2 GPUs per run, for 5 epochs. After all runs, we collect evaluation loss, F1, EM, and runtime to select the best configuration.


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
- Automatic Log Parsing: At the end of all trials, the SLURM script automatically generates a CSV with the same name as the SLURM log `logs/<job_name>-<job_id>.csv`.

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

## Breaking Down the Ray-Tune (ASHA Scheduler) HPO Building Block

### Training Python File

### SLURM Script

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