#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ray_client
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out
#SBATCH --reservation=DS-TRAINING
#SBATCH --gpus=1

module load dl
module load pytorch
module load ray/2.2.0

export jup_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export ip_head=$(cat ./head_node_info | cut -d " " -f 1)
export head_node_ip=$(echo ${ip_head} | cut -d ":" -f 1)
export redis_password=$(cat ./head_node_info | cut -d " " -f 2)
export dashboard_port=$(cat ./head_node_info | cut -d " " -f 3)


ray start --address ${ip_head}  --redis-password ${redis_password} \
	--num-cpus ${SLURM_CPUS_PER_TASK} \
	--block &
sleep 20
ray status --address ${ip_head} --redis_password ${redis_password}

echo "
For dashboard create SSH tunnel via the following \n
ssh -L localhost:${jup_port}:$(/bin/hostname):${jup_port} -L localhost:${dashboard_port}:${head_node_ip}:${dashboard_port} ${USER}@glogin.ibex.kaust.edu.sa
"
jupyter-lab --no-browser --port=${jup_port} --port-retries=0  --ip=$(/bin/hostname)

ray stop
server_jobID=$(squeue -t r -u $USER -n ray_server -O JobID -h)
scancel $server_jobID
