#!/bin/bash
# ───────────────────────────── SLURM DIRECTIVES ─────────────────────────────
#SBATCH --job-name=bloom_hpo_serial_5_epochs   # Job name in queue
#SBATCH --output=logs/%x-%j.out               # Stdout/stderr path: logs/<job>-<id>.out
#SBATCH --nodes=1                             # Single node (all GPUs local)
#SBATCH --gpus-per-node=2                     # 2 × V100 GPUs
#SBATCH --cpus-per-task=6                     # 6 logical CPUs for the job
#SBATCH --time=08:00:00                       # Wall-clock limit (fits 12 runs × 5 epochs)
#SBATCH --constraint=v100                     # Only schedule on V100 partition
#SBATCH --mem=256G                            # System RAM reservation


# ──────────────────────────── PATHS SETUP ─────────────────────────────
CURRENT_DIR=$(pwd)
cd "$CURRENT_DIR/../../"

ROOT_DIR=$(pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/manual"

# ──────────────────────────── ENVIRONMENT SETUP ─────────────────────────────
module load cuda/12.4.1                       # Load matching CUDA toolkit

source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate hpo-raytune                 # Activate conda env with HF + DeepSpeed

# ───────────────────────────── HYPER-PARAM GRID ─────────────────────────────
#   Iterate over every combination of learning-rate (LR), batch-size (BS),
#   and weight-decay (WD). 3 × 2 × 2 = 12 serial fine-tune runs.
LRs=(1e-5 2e-4 5e-6)
BSs=(1 2)
WDs=(0.0 0.01)

# ────────────────────────────── TIME-STAMP LOGS ─────────────────────────────
# Record human-readable start time
printf '\n===== JOB %s START  : %(%F %T %Z)T =====\n' "$SLURM_JOB_ID" -1

# Always record finish time — fires on normal exit, error, or scancel
trap 'printf "\n===== JOB %s FINISH : %(%F %T %Z)T =====\n" "$SLURM_JOB_ID" -1' EXIT

# ────────────────────────── SERIAL GRID EXECUTION ───────────────────────────
# torchrun arguments:
#   --standalone           Run without distributed rendezvous server
#   --nproc_per_node=2     Launch 2 worker processes = 1 per GPU
for lr in "${LRs[@]}"; do
  for bs in "${BSs[@]}"; do
    for wd in "${WDs[@]}"; do
      echo "◆ Running combo: lr=$lr  bs=$bs  wd=$wd"
      torchrun --standalone --nproc_per_node=2 \
        "$SCRIPT_DIR/bloom_hpo_manual.py" \
          --lr "$lr" \
          --bs "$bs" \
          --wd "$wd"
      echo "── combo finished ──"
    done
  done
done

# ──────────────────────────── LOGS PARSING ──────────────────────────────────

python "$SCRIPT_DIR/logs_parser.py" --logs "$CURRENT_DIR/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" --out "$CURRENT_DIR/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.csv"

# NOTE:
# • Each fine-tune run is completely independent; checkpoints go to
#   'checkpoints/' directory defined inside bloom_hpo_manual.py.
# • If you prefer parallel trials, switch to Ray Tune or separate SLURM array jobs.
