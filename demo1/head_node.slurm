#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=ibex-cs

source ~/miniconda3/bin/activate ray_demo
#Requested number of workers
if [ -z ${NUM_WORKERS} ] ; then
  NUM_WORKERS=1
else
  NUM_WORKERS=${NUM_WORKERS}
fi

export server_port=9121
export dashboard_port=9122
export redis_password=${SLURM_JOBID}
export head_node_ip=$(hostname -I | cut -d " " -f 2)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password}" > head_node_info


ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=$HOSTNAME \
        --num-cpus 1 --block &
sleep 20
job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ ))
 do
   job_ids[$i]=$(sbatch -x $SLURM_NODELIST worker_node.slurm | cut -d " " -f 4)
 done 

sleep 20

python -u hello_tune.py --num-samples=100 --max-concurrent-trials=20

# Shutdown workers before the head node
touch $PWD/shutdown.txt
sleep 20
echo " Stopping ray on Head node: $(/bin/hostname)"
ray stop
rm $PWD/shutdown.txt

 
