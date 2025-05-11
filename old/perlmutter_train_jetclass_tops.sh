#!/bin/bash

echo "Starting job $SLURM_JOB_NAME - $SLURM_JOB_ID with the following script:"
echo "----------------------------------------------------------------------------"
cat $0

# Environment
set -x

module load conda
conda activate markov_bridges
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

srun python ../train_jetclass_tops.py
