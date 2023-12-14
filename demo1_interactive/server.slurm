#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --job-name=ray_server
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out
#SBATCH --reservation=DS-TRAINING


module load dl
module load pytorch
module load ray/2.2.0

export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export redis_password=${SLURM_JOBID}


head_node_ip=$(hostname -I | cut -d " " -f 1)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password} ${dashboard_port}" > head_node_info


ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=$HOSTNAME \
        --num-cpus ${SLURM_CPUS_PER_TASK} -vvv --block 
