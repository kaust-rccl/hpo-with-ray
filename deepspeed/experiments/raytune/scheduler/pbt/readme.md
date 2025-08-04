
# Automated HPO with Ray Tune: Population-Based Training (PBT)

## Overview

In this exercise, we perform **automated hyperparameter optimization** using **Ray Tune’s Population-Based Training (PBT)** scheduler.  
Unlike ASHA, which stops bad trials early, **PBT periodically explores new hyperparameter combinations** during training by **cloning and mutating** well-performing trials into weaker ones.

The training is executed via a **Python script** ([`bloom_ray_tune_pbt.py`](./scripts/raytune_hpo/raytune_pbt_scheduler/raytune_pbt_hpo.py))  
and orchestrated on the cluster using:

- **Head node SLURM script** ([`head_node_raytune_pbt_hpo.slurm`](./experiments/raytune_hpo/raytune_pbt_scheduler/head_node_raytune_pbt_hpo.slurm))
- **Worker node SLURM script** ([`worker_node_v100.slurm`](./experiments/raytune_hpo/raytune_pbt_scheduler/worker_node_raytune_pbt_hpo.slurm))

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


### SLURM Script

## Exercise: Launch, Track, and Analyze

### Launching the Jobs

### Result Collection Table

### Quiz Questions
