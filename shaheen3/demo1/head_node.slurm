#!/bin/bash

#SBATCH --partition=shared 

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --time=00:10:00

#SBATCH --job-name=ray_head
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out



module load python
module load pytorch/2.2.1
module list

mkdir -p ${SCRATCH_IOPS}/temp/

#Requested number of workers
if [ -z ${NUM_WORKERS} ] ; then
  NUM_WORKERS=1
else
  NUM_WORKERS=${NUM_WORKERS}
fi

export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export redis_password=${SLURM_JOBID}
export head_node_ip=$(hostname -I | cut -d " " -f 2)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password} ${dashboard_port}" > head_node_info


ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=$HOSTNAME \
        --num-cpus ${SLURM_CPUS_PER_TASK} --block &
sleep 20
job_ids=()

for (( i=1; i<=${NUM_WORKERS}; i++ ))
 do
   job_ids[$i]=$(sbatch -x $SLURM_NODELIST worker_node.slurm | cut -d " " -f 4)
 done 

while [ ! -z $(squeue -n ray_worker -t PD -h -o %A) ]
do
	echo "Waiting for worker(s) to start"
        sleep 30
done
sleep 20

ray status --address ${ip_head} --redis_password ${redis_password}
python -u hello_tune.py --num-samples=50 --max-concurrent-trials=20

# Shutdown workers before the head node
touch $PWD/shutdown.txt
sleep 20
echo " Stopping ray on Head node: $(/bin/hostname)"
ray stop
rm $PWD/shutdown.txt

 
