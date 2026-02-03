#!/bin/bash
#SBATCH --job-name=pbt_head
#SBACTH --reservation=distributedHPO
#SBATCH --output=logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# ----------- dynamic HPO ranges from env vars or defaults ---------------------
: "${LR_LOWER:=5e-6}"
: "${LR_UPPER:=2e-4}"
: "${BS_CHOICES:=1 2}"
: "${WD_CHOICES:=0.0 0.01}"


CURRENT_DIR=$(pwd)
WORKER_DIR=$CURRENT_DIR

ROOT_DIR="$( cd "$CURRENT_DIR/../../../../" && pwd )"

SCRIPT_DIR="$ROOT_DIR/scripts/raytune/scheduler/pbt"
DS_CONFIG_DIR="$ROOt_DIR/config"

module load /sw/rl9g/dl/workshop/modules/hpo-ray

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

export node=$(/bin/hostname -s)

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

sleep 20

# ---------- spawn workers -----------------------------------------------------
job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ )); do
  job_ids[$i]=$(sbatch -x $SLURM_NODELIST "$WORKER_DIR/worker_node_raytune_pbt_hpo.slurm" | awk '{print $4}')
done

# wait until workers leave PD (pending) state
while squeue -n ray_worker_bloom -t PD -h | grep -q . ; do
  echo "Waiting for worker(s) to start"; sleep 20
done

# Define a cleanup function to cancel workers
cleanup() {
  echo "Head job exiting. Cancelling worker jobs..."
  for id in "${job_ids[@]}"; do
    if [[ -n "$id" ]]; then
      scancel "$id"
    fi
  done
}
# Run cleanup on both success and failure
trap cleanup EXIT

# ---------- run the Tune driver ----------------------------------------------
export ip_head
export head_node_ip
export redis_password
echo "Launching script from: $(pwd)"
python -u "$SCRIPT_DIR/raytune_pbt_hpo.py" \
	--num_train_epochs 5 \
	--deepspeed $ROOT_DIR/config/ds_config.json \
	--lr_lower $LR_LOWER \
	--lr_upper $LR_UPPER\
       	--per_device_bs_choices $BS_CHOICES \
	--wd_choices $WD_CHOICES

# ---------- shutdown ----------------------------------------------------------
touch shutdown.txt
sleep 20
ray stop
rm shutdown.txt
