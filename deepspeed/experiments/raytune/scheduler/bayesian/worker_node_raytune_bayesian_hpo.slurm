#!/bin/bash
#SBATCH --job-name=bayesian_worker
#SBACTH --reservation=distributedHPO
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=8
#SBATCH --constraint=v100


# --- ENVIRONMENT SETUP ---
module load dl-workshops hpo-ray

export NUM_CPUS_PER_NODE=${SLURM_CPUS_PER_TASK}
export NUM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}

echo "${NUM_GPUS_PER_NODE}"
read ip_head redis_password dashboard_port < head_node_info
ray start --address ${ip_head}\
          --redis-password ${redis_password} \
          --num-cpus ${NUM_CPUS_PER_NODE} \
          --num-gpus ${NUM_GPUS_PER_NODE} \
          --block &

sleep 20
ray status --address ${ip_head}

# graceful shutdown triggered by the head
while [ ! -f shutdown.txt ]; do sleep 20; done
ray stop
