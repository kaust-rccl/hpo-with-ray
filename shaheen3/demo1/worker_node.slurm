#!/bin/bash

#SBATCH --partition=workq

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread

#SBATCH --time=00:10:00

#SBATCH --job-name=ray_worker
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out



module load python
module load pytorch/2.2.1
module list

mkdir -p ${SCRATCH_IOPS}/temp/


export ip_head=$(cat ./head_node_info | cut -d " " -f 1)
export head_node_ip=$(echo ${ip_head} | cut -d ":" -f 1)
export redis_password=$(cat ./head_node_info | cut -d " " -f 2)
export dashboard_port=$(cat ./head_node_info | cut -d " " -f 3)

ray start --address ${ip_head}  --redis-password ${redis_password} \
	--num-cpus ${SLURM_CPUS_PER_TASK} --block &
sleep 20
ray status --address ${ip_head} --redis_password ${redis_password}
#sleep 10

# worker shutdown strategy
if [ -f "shutdown.txt" ] ; then
  echo " Stopping ray on Node: $(/bin/hostname)"
  ray stop
else
  while [ ! -f "shutdown.txt" ]; 
   do
     sleep 20
   done   
fi
