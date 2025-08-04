# Manual HPO with SLURM:

## Overview

In this exercise, we perform **manual hyperparameter optimization (HPO)** by iterating over a predefined grid of
hyperparameters (learning rate, batch size, weight decay).
**A SLURM script** ([`manual_bloom_hpo.slurm`](./experiments/manual/bloom_hpo_manual.slurm)) handles the **job
submission, environment setup, and looping logic**, while **a Python training script** ([
`manual_bloom_hpo.py`](scripts/manual/bloom_hpo_manual.py)) handles data preprocessing, fine-tuning, and evaluation.

Each hyperparameter combination runs **sequentially** (serial execution), using 2 GPUs per run, for 5 epochs. After all
runs, we collect evaluation loss, and runtime to select the best configuration.

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
  the SLURM log `logs/<job_name>-<job_id>.csv`, that  collects and summarizes key metrics from each hyperparameter combination run, including:
  - **Learning rate**, **batch size**, and **weight decay**
  - **Evaluation loss**, **F1 score**, and **Exact Match (EM)** score
  - **Total training runtime**
  - **GPU-hours used**
  It serves as a performance report across all tried configurations and helps in quickly identifying the best-performing setup for extended training.

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

1. Make sure you are in the same [directory](./) as this `README`, then navigate to the SLURM script directory:
    ```commandline
    cd experiments/manual/
    ```
2. Submit the manual HPO job:
    ```commandline
    sbatch bloom_hpo_manual.slurm
    ```
3. Monitor the job in the queue
    ```commandline
    squeue --me
    ```

### Result Collection Table

- Navigate and open the logs file:
  ```commandline
  cd ./logs
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


