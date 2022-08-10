#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --job-name=ray_server
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out
#SBATCH --account=ibex-cs

source ~/miniconda3/bin/activate ray_demo

export UCX_TLS=tcp,self
export UCX_NET_DEVICES=ib0 

export server_port=9121
export dashboard_port=9122
export redis_password=${SLURM_JOBID}
head_node_ip=$(hostname -I | cut -d " " -f 1)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password}" > head_node_info


ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=$HOSTNAME \
        --num-cpus ${SLURM_CPUS_PER_TASK} -vvv --block 
