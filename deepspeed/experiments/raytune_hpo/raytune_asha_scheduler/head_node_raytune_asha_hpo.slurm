#!/bin/bash
#SBATCH --job-name=ray_head_bloom_5epochs
#SBATCH --output=logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

CURRENT_DIR=$(pwd)
WORKER_DIR=$CURRENT_DIR

cd "$CURRENT_DIR/../../../"

ROOT_DIR=$(pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/raytune_hpo/raytune_asha_scheduler"
DS_CONFIG_DIR="$ROOt_DIR/config"

source /ibex/user/x_mohameta/miniforge/etc/profile.d/conda.sh
conda activate bloom-finetune
module load cuda/12.4.1

# ---------- worker count ------------------------------------------------------
: "${NUM_WORKERS:=1}"     # export NUM_WORKERS before sbatch to change

# ---------- dynamic ports ----------------------------------------------------
export server_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
export dashboard_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
export tensorboard_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)

export node=$(/bin/hostname -s)
echo "
SSH-tunnel for the Ray dashboard:
  ssh -L localhost:${dashboard_port}:${node}:${dashboard_port} \\
      -L localhost:${tensorboard_port}:${node}:${tensorboard_port} \\
      ${USER}@glogin.ibex.kaust.edu.sa
"

# ---------- scratch dirs -----------------------------------------------------
export TB_TMPDIR=$PWD/tboard/${SLURM_JOBID}; mkdir -p $TB_TMPDIR

export redis_password=${SLURM_JOBID}
export head_node_ip=$(hostname -I | awk '{print $2}')
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password} ${dashboard_port}" > head_node_info

# ---------- start Ray head ----------------------------------------------------
ray start --node-ip-address ${head_node_ip} --port ${server_port} \
          --redis-password=${redis_password} --head \
          --dashboard-port ${dashboard_port} --dashboard-host=127.0.0.1 \
          --num-cpus $SLURM_CPUS_PER_TASK  --num-gpus 0 --block &

tensorboard --logdir=${PWD}/logs/${SLURM_JOBID} \
            --port=${tensorboard_port} &

sleep 20

# ---------- spawn workers -----------------------------------------------------
job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ )); do
  job_ids[$i]=$(sbatch -x $SLURM_NODELIST "$WORKER_DIR/worker_node_raytune_asha_hpo.slurm" | awk '{print $4}')
done

# wait until workers leave PD (pending) state
while squeue -n ray_worker_bloom -t PD -h | grep -q . ; do
  echo "Waiting for worker(s) to start"; sleep 20
done

# ---------- run the Tune driver ----------------------------------------------
export ip_head
export head_node_ip
export redis_password
echo "Launching script from: $(pwd)"
python -u "$SCRIPT_DIR/raytune_asha_hpo.py" --deepspeed "$DS_CONFIG_DIR"

# ---------- shutdown ----------------------------------------------------------
touch shutdown.txt
sleep 20
ray stop
rm shutdown.txt
