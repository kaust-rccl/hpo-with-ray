#!/bin/bash
#SBATCH --job-name=ray_tune
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=ibex-cs

source ~/miniconda3/bin/activate ray_demo

export XDG_RUNTIME_DIR=$PWD 
NUM_CPUS=${SLURM_CPUS_PER_TASK}

export server_port=9121
export dashboard_port=9122
export redis_password=${SLURM_JOBID}


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

head_node=${nodes_array[0]}
head_node_ip=$(srun  --nodes=1 --ntasks=1  -w "$head_node" /bin/hostname -I | cut -d " " -f 1)


## STARTING Ray head node
export ip_head=$head_node_ip:${server_port}
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
cmd="srun -u -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${head_node}  \
      ray start --node-ip-address ${head_node_ip} --port ${server_port} \
                --redis-password=${redis_password} --head --num-cpus ${NUM_CPUS}  \
                --dashboard-port ${dashboard_port} --dashboard-host=$HOSTNAME \
                --temp-dir=${HOME}/temp/${SLURM_JOBID} --verbose --block"
echo $cmd
$cmd &

## STARTING Ray worker nodes

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 30

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    cmd="srun -u -w "$node_i" -n 1 -N 1 -c ${NUM_CPUS}  \
          ray start --address "$ip_head" --redis-password=${redis_password} \
                    --num-cpus ${NUM_CPUS}  \
         			--temp-dir=${HOME}/temp/${SLURM_JOBID} --verbose --block"
    echo $cmd
    $cmd &
    
    sleep 40
done

## SUBMIT workload to the Ray cluster
ray status --address ${ip_head} --redis_password ${redis_password} 
sleep 40
python -u hello_tune.py --num-samples=100 --max-concurrent-trials=20
exit 0