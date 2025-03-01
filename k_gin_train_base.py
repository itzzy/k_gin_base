import argparse
import numpy as np
import yaml
import torch
import os
from utils import wandb_setup, dict2obj
# from trainer import TrainerKInterpolator
from k_gin_trainer_base import TrainerKInterpolator
import torch.distributed as dist

# 设置环境变量来确保 W&B 以离线模式运行
# import os
# os.environ["WANDB_MODE"] = "offline"


# 设置环境变量 CUDA_VISIBLE_DEVICES
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'  # 指定使用 GPU 7 和 GPU 3
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 指定使用 GPU 4 和 GPU 6  
#0和3
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,5,7' 0-5(nvidia--os) 2-6 3-7
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,7'


# PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
os.environ['OMP_NUM_THREADS'] = '1'
# 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, help='config file (.yml) containing the hyper-parameters for inference.')
parser.add_argument('--debug', action='store_true', help='if true, model will not be logged and saved')
parser.add_argument('--seed', type=int, help='seed of torch and numpy', default=1)
parser.add_argument('--val_frequency', type=int, help='training data and weights will be saved in this frequency of epoch')

if __name__ == '__main__':

    args = parser.parse_args()

    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        print(f'Using {args.config} as config file')
        config = yaml.load(f, Loader=yaml.FullLoader)

    for arg in vars(args):
        if getattr(args, arg):
            if arg in config['general']:
                print(f'Overriding {arg} from argparse')
            config['general'][arg] = getattr(args, arg)

    if not config['general']['debug']: wandb_setup(config)
    # if not config['general']['debug']:
    #     print("Initializing Wandb in offline mode...")
    #     wandb.init(mode="offline")
    #     print("Wandb initialized in offline mode.")
    
    config = dict2obj(config)
    print('config:',config)
    print('config.general.debug:',config.general.debug)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.general.gpus}'

    config.general.debug = args.debug
    print('config.general.debug:',config.general.debug)
    # trainer = TrainerKInterpolator(config)
    trainer = TrainerKInterpolator(config, local_rank)
    print("Starting trainer run method...")
    trainer.run()
    print("Trainer run method completed.")

    # 销毁分布式环境
    dist.destroy_process_group()
    
# 使用 torch.distributed.launch 启动分布式训练：
# python -m torch.distributed.launch --nproc_per_node=2 --use_env k-gin-git/train.py --config your_config.yml
# python -m torch.distributed.launch --nproc_per_node=2 --use_env k-gin-git/train.py --config config.yml
'''
LOCAL_RANK 是一个环境变量，用于指示当前进程在本地机器上的 GPU 设备编号。在分布式训练中，LOCAL_RANK 通常由分布式启动器（如 torch.distributed.launch）自动设置，用于确保每个进程使用不同的 GPU。
详细解释
LOCAL_RANK：表示当前进程在本地机器上的 GPU 设备编号。例如，如果你有两块 GPU，并且使用 torch.distributed.launch 启动两个进程，那么 LOCAL_RANK 的值将分别为 0 和 1。
RANK：表示当前进程在整个分布式训练中的全局编号。它是一个全局唯一的编号，用于标识每个进程。
使用 LOCAL_RANK
在分布式训练中，你可以使用 LOCAL_RANK 来设置当前进程使用的 GPU 设备。例如：
import os
import torch
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 获取本地进程的 GPU 设备编号
local_rank = int(os.environ['LOCAL_RANK'])

# 设置当前进程使用的 GPU 设备
torch.cuda.set_device(local_rank)

# 其他初始化代码
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
'''