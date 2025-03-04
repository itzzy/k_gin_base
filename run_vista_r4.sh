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

# /nfs/zzy/code/k_gin_base/train_kgin_base_vista_r8.py /nfs/zzy/code/k_gin_base/config_kgin_base_vista_r8.yaml
python train_kgin_base_vista_r4.py --config config_kgin_base_vista_r4.yaml

# out_kgin_vista_r4_0222.npy
# test_kgin_vista_r4
# /data0/zhiyong/data/data222/base/r4
#scp experiments/test_kgin_vista_r4/model_281.pth  experiments/test_kgin_vista_r4/model_300.pth out_kgin_vista_r4_0222.npy zhiyongzhang@172.20.35.37:/data0/zhiyong/data/data222/base/r4
#scp experiments/test_kgin_vista_r6/model_281.pth  experiments/test_kgin_vista_r6/model_300.pth out_kgin_vista_r6_0222.npy zhiyongzhang@172.20.35.37:/data0/zhiyong/data/data222/base/r6

# out_kgin_kv_vista_r4_0222.npy
# test_kgin_kv_vista_r4
# /data0/zhiyong/data/data222/kv/r4
#scp experiments/test_kgin_kv_vista_r4/model_281.pth  experiments/test_kgin_kv_vista_r4/model_300.pth out_kgin_kv_vista_r4_0222.npy zhiyongzhang@172.20.35.37:/data0/zhiyong/data/data222/kv/r4
#scp experiments/test_kgin_kv_vista_r6/model_281.pth  experiments/test_kgin_kv_vista_r6/model_300.pth out_kgin_kv_vista_r6_0222.npy zhiyongzhang@172.20.35.37:/data0/zhiyong/data/data222/kv/r6