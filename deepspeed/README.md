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

#### Bayesian (Bayesian Optimization with HyperBand - BOHB)

- Combines **Bayesian Optimization** (learns from past trials to suggest better hyperparameters) with **HyperBand** (
  efficient early-stopping strategy).

- Starts with many short trials and **promotes the best ones** for longer training, refining them with **probabilistic
  guidance**.

- Best used when you have a **limited GPU budget** and want to balance **smart search** with **efficient resource use**.

- Unlike ASHA, BOHB doesn’t just explore randomly — it builds a **model of the search space** and chooses configurations
  based on **predicted performance**.

**Quick Comparison**

- ASHA → Random sampling + early stopping.

- PBT → Dynamic evolution of hyperparameters during training.

- Bayesian → **Model-based search** with early stopping — efficient for **small budgets or expensive trials**.

## DeepSpeed

### Purpose and Role in Training

[**DeepSpeed**](https://www.deepspeed.ai/) is an optimization library designed for **scaling and accelerating deep
learning training**, especially for large models like BLOOM.

In this workshop, DeepSpeed is used to:

- **Reduce memory usage** via ZeRO Stage 3 optimization.
- **Enable mixed precision training** (`fp16`) for faster computation and lower memory.
- **Automatically scale** batch sizes to maximize GPU utilization.
- **Train large models** efficiently on limited hardware (e.g., 1–2 GPUs).

It integrates seamlessly with Hugging Face’s `Trainer` and requires only a config file — no modification to training
code is needed.


> For a full breakdown of the DeepSpeed config used in this workshop, see  
> [`experiments/deepspeed/README.md`](./config/README.md)

---

## Environment Setup and File Structure

### Environment Setup

<!-- TODO: yml file and instruction -->

### Project Structure

This repository is organized into modular directories for code, configuration, and experiments.

```plaintext
.
├── deepspeed/
│   ├── config/                  # DeepSpeed configuration files
│   │   ├── ds_config.json       # ZeRO-3 + FP16 training config
│   │   └── README.md            # Explanation of the config fields
│
├── experiments/                # SLURM job scripts and run setups
│   ├── manual/                 # Manual grid search HPO
│   │   ├── bloom_hpo_manual.slurm
│   │   └── README.md
│   └── raytune/
│       ├── scheduler/
│       │   ├── asha/           # ASHA-based Ray Tune setup
│       │   │   ├── head_node_raytune_asha_hpo.slurm
│       │   │   ├── worker_node_raytune_asha_hpo.slurm
│       │   │   └── README.md
│       │   ├── bayesian/       # BOHB setup (Bayesian Optimization with HyperBand)
│       │   │   └── README.md
│       │   └── pbt/            # Population-Based Training setup
│       │       └── README.md
│       └── README.md           # Ray Tune general overview
│
├── scripts/                    # Python training scripts
│   ├── manual/
│   │   ├── bloom_hpo_manual.py # Runs single grid search config
│   │   └── logs_parser.py      # Parses manual run logs into CSV
│   └── raytune/
│       ├── scheduler/
│       │   ├── asha/raytune_asha_hpo.py
│       │   ├── bayesian/README.md
│       │   └── pbt/README.md
│       └── README.md           # Ray Tune script overview
│
└── README.md                   # Main workshop overview and grouping instructions
```

---

# Team Grouping & HPO Assignment Instructions

In this workshop, you'll work in **teams of 3 students**. Each group will:

1. Choose a **hyperparameter range** for:
    - **Learning Rate (lr)**
    - **Weight Decay (wd)**
    - **Batch Size (bs)**

2. Divide up the HPO strategies as follows:
    - **Member 1:** Automated HPO with **ASHA Scheduler**
    - **Member 2:** Automated HPO with **Population-Based Training (PBT)**
    - **Member 3:** Automated HPO with **Bayesian Training (BOHB)**

3. Run the experiments using your assigned method.

4. At the end, **collect results**, compare them as a team, and fill in the provided **group summary**.

---

# Navigation to Instructions

Each member should now navigate to the README for their assigned method:

| Method               | Path                                                                         | Instructions                                                            |
|----------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Manual HPO (pre-run) | [`experiments/manual/`](./experiments/manual/)                               | [Instructions Here](./experiments/manual/readme.md)                     |
| ASHA (Ray Tune)      | [`experiments/raytune/asha/`](./experiments/raytune/scheduler/asha/)         | [Instructions Here](./experiments/raytune/scheduler/asha/readme.md)     |
| PBT (Ray Tune)       | [`experiments/raytune/pbt/`](./experiments/raytune/scheduler/pbt/)           | [Instructions Here](./experiments/raytune/scheduler/pbt/readme.md)      |
| Bayesian (Ray Tune)  | [`experiments/raytune/bayesian/`](./experiments/raytune/scheduler/bayesian/) | [Instructions Here](./experiments/raytune/scheduler/bayesian/readme.md) |

---

# Group Submission Checklist

Each group must submit the following:

- [ ] A filled **results table** from each method.
- [ ] **Quiz answers** from each scheduler's README.
- [ ] A 5–7 line **comparison** discussing:
    - Which method found the best configuration?
    - Which used fewer GPU-hours?
    - Which was faster overall?
    - What would you use for real-world tuning?

### Cost Comparison (Fill-in Template)

You can use this format to **summarize and compare results across methods**, and to justify your preferred tuning
strategy.

| **Run Type**        | **Eval Loss (30 Epochs)** | **Runtime (to find best HP)** | **# GPUs** | **GPU Minutes** | **Cost Ratio (Ray/Manual)** |
|---------------------|---------------------------|-------------------------------|------------|-----------------|-----------------------------|
| Manual Best         | 11.7463                   | 177                           | 2          | 354             | 1        (reference)        |
| Ray Best (ASHA)     |                           |                               |            |                 |                             |
| Ray Best (PBT)      |                           |                               |            |                 |                             |
| Ray Best (Bayesian) |                           |                               |            |                 |                             |

> Note: Cost ratio is based on total GPU time consumed to find the best configuration (e.g.,
`Ray GPU-minutes / Manual GPU-minutes`).
