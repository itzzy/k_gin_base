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

# export CUDA_VISIBLE_DEVICES=0

wandb offline

# /nfs/zzy/code/k_gin_base/train_kgin_base_vista_r8.py /nfs/zzy/code/k_gin_base/config_kgin_base_vista_r8.yaml
python train_kgin_base_r4.py --config config_kgin_base_r4.yaml