# Introduction

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

---

