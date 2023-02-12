export WANDB_MODE=offline
export WANDB_DIR=$PWD/logs/${EXPERIMENT}/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=ray_wb_${EXPERIMENT}


