# HPO-with-ray

Welcome to **HPO-with-ray** — a hands-on repository used in the hyperparameter optimization (HPO) training sessions
using ray framework by the
KAUST Supercomputing Core Lab (KSL).

The end goal is to help you understand:

- **how** to use ray framework for hpo
- **when** to use each hpo technique

---

## Quick start

1. **Complete the prerequisites**

   Start with the instructions in [`prerequisites/`](./prerequisites)

   These steps **must be completed before the workshop starts**.


2. **Submit jobs early using the Job Submission Manual** (during workshop)
    - For **faster and smoother workshops**, each major module includes a
      **Job Submission Manual** designed to help you:
        - submit *all* experiment jobs at the beginning of the session
        - allow jobs time to queue and run while the workshop continues
        - have results ready for analysis later
    - You will find a Job Submission Manual inside:
        - [`deepspeed/`](./deepspeed)

   **At the beginning of the workshop, open the module’s Job Submission Manual and submit all jobs first.**

---

## Repository navigation

### Training modules

Each directory below represents a self-contained learning module:

- [`deepspeed/`](./deepspeed)

  Core HPO training examples.
  Demonstrates Hyperparameter Optimization (HPO) for fine-tuning large language models. You will experiment with both
  manual and automated approaches to explore how different hyperparameters affect model performance and training cost.

- [`demo1/`](./demo1)

  A simple demonstration for running a ray process using only cpu resources

- [`demo2/`](./demo2)
  A simple demonstration showing how to use ray to train a simple CNN network for identifiying written digits using
  MNIST dataset

- [`demo3/`](./demo3)

  A simple demonstration with same model and dataset as demo2 but using the pbt (population based trainning) technique.

- [`demo4/`](./demo4)

  A simple demonstration with same model and dataset as demo2 but using random search technique

- [`shaheen3/`](./shaheen3/)
  A simple demonstration with exactly the same objective of demo1 but meant to run in shaheen3 cluster

---

## What to expect inside each module

While details vary per framework, most modules contain:

- Runnable training scripts
- SLURM job scripts for single-node and multi-node runs
- Module-specific README files with usage instructions

Each module is designed to be runnable independently once prerequisites are completed.
