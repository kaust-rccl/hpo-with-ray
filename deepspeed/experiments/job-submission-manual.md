# Jobs Submission Manual

## Purpose of This Guide

This guide is designed to help workshop participants **submit all HPO experiment jobs early in the session**, allowing
sufficient time for jobs to queue, start, and complete while the workshop progresses.



---

## Working Directory Context

All commands in this guide assume that you start from the [experiments/](.) directory.

```commandline
cd experiments
```

---

## Track Your Submissions

You can check the status of your submitted jobs at any time using:

```commandline
squeue --me
```

### What this shows

- Job ID

- Job name

- State (PENDING, RUNNING, COMPLETED)

- Elapsed time

- Time limit

- Number of nodes

- Reason / assigned node

---

## 1) ASHA

This section corresponds to
Exercise:  [ASHA: Exercise: Launch, Track, and Analyze](./raytune/scheduler/asha/readme.md#exercise-launch-track-and-analyze)

```commandline
cd raytune/scheduler/asha
sbatch head_node_raytune_asha_hpo.slurm
cd - 
```

---

## 2) PBT

This section corresponds to
Exercise:  [PBT: Exercise: Launch, Track, and Analyze](./raytune/scheduler/pbt/readme.md#exercise-launch-track-and-analyze)

```commandline
cd raytune/scheduler/pbt
sbatch head_node_raytune_pbt_hpo.slurm
cd - 
```

---

## 3) Bayesian

This section corresponds to
Exercise:  [Bayesian: Exercise: Launch, Track, and Analyze](./raytune/scheduler/bayesian/readme.md#exercise-launch-track-and-analyze)

```commandline
cd raytune/scheduler/bayesian
sbatch head_node_raytune_bayesian_hpo.slurm
cd -
```
---

## Conclusion

After submitting the jobs, participants are expected to **periodically check their job status** using the provided SLURM
commands.

Once the workshop reaches the results analysis phase, please return to the **generated log files and outputs**
corresponding to each experiment. These logs will be used to:

- Analyze performance and scaling behavior
- Discuss trade-offs observed across single-GPU, multi-GPU, and multi-node runs

Having jobs submitted early and tracking their progress ensures that **meaningful results are available for discussion**
by the end of the session.

