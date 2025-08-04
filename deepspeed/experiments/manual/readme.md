# Manual HPO with SLURM:

## Overview

In this exercise, we perform **manual hyperparameter optimization (HPO)** by iterating over a predefined grid of
hyperparameters (learning rate, batch size, weight decay).
**A SLURM script** ([`bloom_hpo_manual.slurm`](./bloom_hpo_manual.slurm)) handles the **job
submission, environment setup, and looping logic**, while **a Python training script** ([
`bloom_hpo_manual.py`](./../../scripts/manual/bloom_hpo_manual.py)) handles data preprocessing, fine-tuning, and evaluation.

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
  the SLURM log `logs/<job_name>-<job_id>.csv`, that collects and summarizes key metrics from each hyperparameter
  combination run, including:
    - **Learning rate**, **batch size**, and **weight decay**
    - **Evaluation loss**, **F1 score**, and **Exact Match (EM)** score
    - **Total training runtime**
    - **GPU-hours used**
      It serves as a performance report across all tried configurations and helps in quickly identifying the
      best-performing setup for extended training.

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

##  Manual HPO Results Summary

The manual HPO experiment was pre-run on a single node with 2 GPUs using a grid of 12 hyperparameter combinations. Each trial ran sequentially for 5 epochs, and the evaluation loss and runtime were recorded.

Below is the summary of the results:

### Manual HPO Results

| #  | Learning Rate (`lr`) | Batch Size (`bs`) | Weight Decay (`wd`) | Eval Loss          | Runtime (s) |
|----|----------------------|-------------------|---------------------|--------------------|-------------|
| 1  | 1e-05                | 1                 | 0.0                 | 9.7208             | 993.66      |
| 2  | 1e-05                | 1                 | 0.01                | 9.7208             | 1010.88     |
| 3  | 1e-05                | 2                 | 0.0                 | 10.0043            | 645.03      |
| 4  | 1e-05                | 2                 | 0.01                | 10.0043            | 639.84      |
| 5  | 0.0002               | 1                 | 0.0                 | 30.2203            | 1205.72     |
| 6  | 0.0002               | 1                 | 0.01                | 30.2203            | 1224.02     |
| 7  | 0.0002               | 2                 | 0.0                 | 21.1526            | 761.29      |
| 8  | 0.0002               | 2                 | 0.01                | 21.1526            | 779.51      |
| 9  | 5e-06                | 1                 | 0.0                 | 9.3347             | 1031.78     |
| 10 | 5e-06                | 1                 | 0.01                | 9.3347             | 1003.58     |
| 11 | 5e-06                | 2                 | 0.0                 | 9.5698             | 570.32      |
| 12 | 5e-06                | 2                 | 0.01                | 9.5698             | 553.48      |

> All runs were performed using the same SQuAD subset, model configuration, and DeepSpeed setup for fair comparison.

---

### Observations

- The best configuration in terms of evaluation loss: **Trial #9** (lr = `5e-06`, bs = `1`, wd = `0.0`) with loss `9.3347`.
- The fastest run: **Trial #12** with runtime `553.48` seconds.
- The HPO experiment consumed a total runtime of  `2 hours, 57 minutes`

You’ll use these results later to compare against Ray Tune’s automated HPO methods.

## Long-Run Training with Manual Best (30 Epochs)

After selecting the best configuration from above, we reran the model using:

- `lr = 5e-06`, `bs = 1`, `wd = 0.0`
- Full 30 epochs of fine-tuning

This simulates a long training job after tuning is complete.

| **Metric**         | **Value**                        |
|--------------------|----------------------------------|
| Final Eval Loss    | 11.7463                          |
| Total GPU Time     | 1 hour, 31 minutes, and 8.3 sec  |