#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --reservation=DS-TRAINING

module load dl
module load pytorch
module load ray/2.2.0

#Requested number of workers
if [ -z ${NUM_WORKERS} ] ; then
  NUM_WORKERS=1
else
  NUM_WORKERS=${NUM_WORKERS}
fi



export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export tensorboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export node=$(/bin/hostname -s)
echo "
Connect to dashboard by creating SSH tunnels. Copy the following command in a new terminal
and connect to localhost via your browser.
ssh -L localhost:${dashboard_port}:${node}:${dashboard_port} -L localhost:${tensorboard_port}:${node}:${tensorboard_port} ${USER}@glogin.ibex.kaust.edu.sa
"


export TB_TMPDIR=$PWD/tboard/${SLURM_JOBID}
mkdir -p ${TB_TMPDIR}

export redis_password=${SLURM_JOBID}
export head_node_ip=$(hostname -I | cut -d " " -f 2)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password} ${dashboard_port}" > head_node_info


ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=127.0.0.1 \
        --num-cpus 1 --block &
tensorboard --logdir=${PWD}/logs/${SLURM_JOBID} --port=${tensorboard_port} & 
sleep 20

job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ ))
 do
   job_ids[$i]=$(sbatch -x $SLURM_NODELIST worker_node.slurm | cut -d " " -f 4)
 done 

while [ ! -z $(squeue -n ray_worker -t PD -h -o %A) ]
do
	echo "Waiting for worker(s) to start"
        sleep 20
done


python -u ray_mnist_pytorch.py --use-gpu \
          --cpus-per-trial=4 --gpus-per-trial=1 \
          --num-samples=200 \
          --max-concurrent-trials=32 \
          --resume --logs-dir=${PWD}/logs/${RESUME_JOBID}
          


# Shutdown workers before the head node
touch $PWD/shutdown.txt
sleep 20
echo " Stopping ray on Head node: $(/bin/hostname)"
ray stop
rm $PWD/shutdown.txt

 
