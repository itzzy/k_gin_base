#!/bin/bash
#SBATCH -N 1                            # the number of node
#SBATCH -p gpu_part                     # partition
#SBATCH -n 1                            # the number of process 
#SBATCH --gres=gpu:1                    # format: gpu:n, n is the number of gpu you request
#SBATCH --job-name=zzy_train                # job name
#SBATCH --time=240:00:00                # time limit hrs:min:sec
#SBATCH --mem=300G                      # memory limit MB

# activate conda envs
source /public/home/macong/anaconda3/bin/activate zzy

# load package
module load compiler/gcc/11.2.0 
module load cuda/cuda_12.2/12.2 

# your program

which python3
which nvcc
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

wandb offline

python train_dcrnn_r4_test.py --config config_dcrnn_r4_test_ma.yaml