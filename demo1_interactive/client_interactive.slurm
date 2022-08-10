#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ray_client
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out
#SBATCH --account=ibex-cs

source ~/miniconda3/bin/activate ray
export UCX_TLS=tcp,self
export UCX_NET_DEVICES=ib0 

export dashboard_port=9122

export jup_port=10101
export ip_head=$(cat ./head_node_info | cut -d " " -f 1)
export head_node_ip=$(echo ${ip_head} | cut -d ":" -f 1)
export redis_password=$(cat ./head_node_info | cut -d " " -f 2)
ray start --address ${ip_head}  --redis-password ${redis_password} \
	--num-cpus ${SLURM_CPUS_PER_TASK} \
	--block &
sleep 20
ray status --address ${ip_head} --redis_password ${redis_password}
echo "For dashboard create SSH tunnel via the following \n
ssh -L ${jup_port}:$(/bin/hostname):${jup_port} -L ${dashboard_port}:${head_node_ip}:${dashboard_port} ${USER}@ilogin.ibex.kaust.edu.sa"

jupyter notebook --no-browser --port=${jup_port} --ip=$(hostname -s) --NotebookApp.token='' --NotebookApp.password=''
ray stop
server_jobID=$(squeue -t r -u $USER -n ray_server -O JobID -h)
scancel $server_jobID
