import os
import sys
import pathlib
import torch
import glob
import tqdm
import time
from torch.utils.data import DataLoader
from dataset.dataloader import CINE2DT
from model.k_interpolator import KInterpolator
from losses import CriterionKGIN
from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, NativeScalerWithGradNormCount as NativeScaler, add_weight_decay
from utils import multicoil2single

import numpy as np
import datetime

# PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
os.environ['OMP_NUM_THREADS'] = '1'
# 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 指定使用 GPU 1 和 GPU 4

# 设置环境变量 CUDA_VISIBLE_DEVICES  0-5(nvidia--os) 2-6 3-7
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'  # 指定使用 GPU 7 和 GPU 3
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 指定使用 GPU 4 和 GPU 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# nohup python train_kgin_base_r8.py --config config_kgin_base.yaml > log_0107_test.txt 2>&1 &

class TrainerAbstract:
    def __init__(self, config):
        print("TrainerAbstract initialized.")
        super().__init__()
        self.config = config.general
        self.debug = config.general.debug
        if self.debug: config.general.exp_name = 'test_kgin_r12'
        self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.only_infer = config.general.only_infer
        self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1

        # data
        train_ds = CINE2DT(config=config.data, mode='train')
        # train_ds = CINE2DT(config=config.data, mode='val')
        test_ds = CINE2DT(config=config.data, mode='val')
        self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
                                    pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_ds, num_workers=2, drop_last=False, batch_size=1, shuffle=False)

        # network
        self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
        self.network.initialize_weights()
        self.network.cuda()
        print("Parameter Count: %d" % count_parameters(self.network))

        # optimizer
        param_groups = add_weight_decay(self.network, config.training.optim_weight_decay)
        self.optimizer = eval(f'torch.optim.{config.optimizer.which}')(param_groups, **eval(f'config.optimizer.{config.optimizer.which}').__dict__)

        # if config.training.restore_ckpt: self.load_model(config.training)
        # 判断配置（config）中的 training.restore_ckpt 属性是否为 True。
        # 如果是 True，表示希望从之前保存的检查点恢复模型，那么就会调用 self.load_model 方法，
        # 并传入 config.training 作为参数，启动恢复模型的相关操作。
        if config.training.restore_training: self.load_model(config.training)
        self.loss_scaler = NativeScaler()

    # def load_model(self, args):

    #     if os.path.isdir(args.restore_ckpt):
    #         args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
    #     ckpt = torch.load(args.restore_ckpt)
    #     self.network.load_state_dict(ckpt['model'], strict=True)

    #     print("Resume checkpoint %s" % args.restore_ckpt)
    #     if args.restore_training:
    #         self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #         self.start_epoch = ckpt['epoch'] + 1
    #         # self.loss_scaler.load_state_dict(ckpt['scaler'])
    #         print("With optim & sched!")
    def load_model(self, args):
        if os.path.isdir(args.restore_ckpt):
            # args.restore_ckpt = max(glob.glob(f'{args.resture_ckpt}/*.pth'), key=os.path.getmtime)
            args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
        ckpt = torch.load(args.restore_ckpt)
        self.network.load_state_dict(ckpt['model'], strict=True)
        self.start_epoch = ckpt['epoch'] + 1
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler' in ckpt and hasattr(self, 'loss_scaler'):
            self.loss_scaler.load_state_dict(ckpt['scaler'])
        print("Resume checkpoint %s" % args.restore_ckpt)

    def save_model(self, epoch):
        ckpt = {'epoch': epoch,
                'model': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scaler': self.loss_scaler.state_dict()
                }
        torch.save(ckpt, f'{self.experiment_dir}/model_{epoch+1:03d}.pth')


class TrainerKInterpolator(TrainerAbstract):

    def __init__(self, config):
        print("TrainerKInterpolator initialized.")
        super().__init__(config=config)
        self.train_criterion = CriterionKGIN(config.train_loss)
        self.eval_criterion = CriterionKGIN(config.eval_loss)
        self.logger = Logger()
        self.scheduler_info = config.scheduler

    def run(self):
        print("Starting run method")
        # 数据加载
        print("Loading data")
        # 模型初始化
        print("Initializing model")
        # 训练循环
        print("Starting training loop")
        pbar = tqdm.tqdm(range(self.start_epoch, self.num_epochs))
        for epoch in pbar:
            self.logger.reset_metric_item()
            start_time = time.time()
            if not self.only_infer:
                self.train_one_epoch(epoch)
            self.run_test()
            self.logger.update_metric_item('train/epoch_runtime', (time.time() - start_time)/60)
            # if epoch % self.config.weights_save_frequency == 0 and not self.debug and epoch > 150:
            if epoch % self.config.weights_save_frequency == 0:
                self.save_model(epoch)
            if epoch == self.num_epochs - 1:
                self.save_model(epoch)
            if not self.debug:
                self.logger.wandb_log(epoch)

    def train_one_epoch(self, epoch):
        start_time = time.time()
        # 累计损失
        running_loss = 0.0
        self.network.train()
        for i, (kspace, coilmaps, sampling_mask) in enumerate(self.train_loader):
            kspace,coilmaps,sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
            # train_one_epoch-kspace torch.Size([4, 20, 18, 192, 192])
            # 18对应t 20-线圈数 4-batc_size
            # print('train_one_epoch-kspace', kspace.shape)
            # train_one_epoch-coilmaps torch.Size([4, 20, 1, 192, 192])
            # print('train_one_epoch-coilmaps', coilmaps.shape)
            # train_one_epoch-sampling_mask torch.Size([4, 18, 192])
            # print('train_one_epoch-sampling_mask', sampling_mask.shape)
            ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
            # train_one_epoch-ref_kspace torch.Size([4, 18, 192, 192])
            # print('train_one_epoch-ref_kspace', ref_kspace.shape)
            # train_one_epoch-ref_kspace-dtype: torch.complex64
            # print('train_one_epoch-ref_kspace-dtype:', ref_kspace.dtype)
            # train_one_epoch-ref_img torch.Size([4, 18, 192, 192])
            # print('train_one_epoch-ref_img', ref_img.shape)
            # train_one_epoch-ref_img-dtype: torch.complex64
            # print('train_one_epoch-ref_img-dtype:', ref_img.dtype)
            # kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2) #[1,18,1,192]
            kspace = ref_kspace
            # kspace_real = c2r(kspace)
            # train_one_epoch-kspace_real torch.Size([2, 36, 192, 192])
            # train_one_epoch-kspace_real torch.Size([2, 192, 18, 192, 2])
            # print('train_one_epoch-kspace_real', kspace_real.shape)
            # train_one_epoch-kspace_real-dtype: torch.float32
            # print('train_one_epoch-kspace_real-dtype:', kspace_real.dtype)

            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            with torch.cuda.amp.autocast(enabled=False):
                k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
                # AttributeError: 'list' object has no attribute 'shape'
                # print('train_one_epoch-k_recon_2ch', k_recon_2ch.shape)
                # train_one_epoch-im_recon torch.Size([4, 18, 192, 192])
                # print('train_one_epoch-im_recon', im_recon.shape)
                sampling_mask = sampling_mask.repeat_interleave(ref_kspace.shape[2], 2)
                # train_one_epoch-sampling_mask-2 torch.Size([4, 18, 36864])
                # print('train_one_epoch-sampling_mask-2', sampling_mask.shape)
                ls = self.train_criterion(k_recon_2ch, torch.view_as_real(ref_kspace), im_recon, ref_img, kspace_mask=sampling_mask)

                self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())

            # 使用 reduce 将每个进程的损失值聚合到主进程
            loss_reduced = ls['k_recon_loss_combined']
            
            running_loss += loss_reduced.item()
            # 添加打印信息
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            eta = datetime.timedelta(seconds=int((elapsed_time / (i + 1)) * (len(self.train_loader) - (i + 1))))
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            # 更新tqdm显示信息
            # Log the detailed information
            if i % 50 ==0:
                print(
                    f"Epoch: [{epoch}] [{i + 1}/{len(self.train_loader)}] eta: {str(eta)} "
                    f"lr: {current_lr:.6f} loss: {loss_reduced.item():.4f} ({running_loss / (i + 1):.4f}) "
                    f"time: {elapsed_time / (i + 1):.4f} data: 0.0002 max mem: {max_memory:.0f}"
                )
            
            torch.cuda.empty_cache()
            self.logger.update_metric_item('train/k_recon_loss', ls['k_recon_loss'].item()/len(self.train_loader))
            self.logger.update_metric_item('train/recon_loss', ls['photometric'].item()/len(self.train_loader))

    def run_test(self):
        out = torch.complex(torch.zeros([118, 18, 192, 192]), torch.zeros([118, 18, 192, 192])).to(device)
        # torch.Size([118, 18, 192, 192]) torch.complex64
        # print('run_test-ooutt-shape:',out.shape, out.dtype)
        self.network.eval()
        psnr_values = []  # 新增：用于收集所有PSNR值的列表
        with torch.no_grad():
            for i, (kspace, coilmaps, sampling_mask) in enumerate(self.test_loader):
                kspace,coilmaps,sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
                ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
                # kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2)
                # torch.Size([2, 18, 192, 192])
                # print('ref_kspace-shape:',ref_kspace.shape)
                
                # np.save('out_1130_2.npy', out)
                kspace = ref_kspace

                k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask) # size of kspace and mask: [B, T, H, W]
                k_recon_2ch = k_recon_2ch[-1]

                kspace_complex = torch.view_as_complex(k_recon_2ch)
                #  torch.Size([2, 18, 192, 192])
                # print('kspace_complex-shape:',kspace_complex.shape)
                sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)
                
                out[i] = kspace_complex

                ls = self.eval_criterion([kspace_complex], ref_kspace, im_recon, ref_img, kspace_mask=sampling_mask, mode='test')
                # 收集每个样本的PSNR值
                psnr_values.append(ls['psnr'].item())  # 修改：记录原始PSNR值
                self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))
            # 计算统计量 均值和方差
            psnr_mean = np.mean(psnr_values)
            psnr_var = np.var(psnr_values)
            # 打印结果
            print(f'\nkgin_base_r12 Validation PSNR - Mean: {psnr_mean:.4f} ± {np.sqrt(psnr_var):.4f} | Variance: {psnr_var:.4f}')
            
            print('...', out.shape, out.dtype)
            out = out.cpu().data.numpy()
            # np.save('out.npy', out)
            # np.save('out_1120.npy', out)
            # np.save('out_1130_3.npy', out)
            # np.save('out_kgin_base_0108.npy', out)
            # 尝试保存数组到文件，如果文件已存在则覆盖
            try:
                np.save('out_kgin_base_0319_r12.npy', out)
            except OSError as e:
                print(f"An error occurred: {e}")
            self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
            self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr'])


def c2r(kspace):
    """
    将复数形式的kspace张量转换为五维实数形式张量，新增第二个维度（大小为2）来分别表示实部和虚部。

    参数:
    kspace (torch.Tensor): 复数形式的张量，形状为 (batch_size, time_steps, height, width)，数据类型为torch.complex64等复数类型

    返回:
    torch.Tensor: 转换后的实数形式张量，形状为 (batch_size, 2, time_steps, height, width)，数据类型为torch.float32
    """
    # 使用torch.view_as_real将复数张量转换为实部和虚部的表示形式
    # 结果的形状变为 (batch_size, time_steps, 2, height, width)，其中最后一个维度的2表示实部和虚部
    kspace_real_imag = torch.view_as_real(kspace)

    # 调整维度顺序，将表示实部和虚部的维度放到第二个维度，保持batch_size等其他维度顺序
    # 转换后的形状变为 (batch_size, 2, time_steps, height, width)
    # kspace_real_imag = kspace_real_imag.permute(0, 2, 1, 3, 4)
    kspace_real_imag = kspace_real_imag.permute(0, 4, 1,2,3)

    return kspace_real_imag

# import os
# import sys
# import pathlib
# import torch
# import glob
# import tqdm
# import time
# from torch.utils.data import DataLoader
# from dataset.dataloader import CINE2DT
# from model.k_interpolator import KInterpolator
# from losses import CriterionKGIN
# from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, NativeScalerWithGradNormCount as NativeScaler, add_weight_decay
# from utils import multicoil2single
# import numpy as np
# import datetime


# import wandb
# import json

# import torch.nn.parallel
# import torch.utils.data.distributed

# # from wandb.util import WandBHistoryJSONEncoder

# # class MyEncoder(WandBHistoryJSONEncoder):
# #     def default(self, obj):
# #         if isinstance(obj, complex):
# #             return {"real": obj.real, "imag": obj.imag}
# #         return super().default(obj)

# # os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
# # os.environ["CUDA_VISIBLE_DEVICES"] = "4" #,0,1,2,4,5,6,7
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# # print(f"Using device: {device}")
# # print(f"Actual GPU being used: torch.cuda.current_device() -> {torch.cuda.current_device()}")
# # print(f"GPU name: {torch.cuda.get_device_name(device)}")

# class TrainerAbstract:
#     def __init__(self, config,local_rank):
#         print('TrainerAbstract1')
#         super().__init__()
#         print('TrainerAbstract2')
#         # config.general.debug: False
#         self.config = config.general
#         self.debug = config.general.debug
#         if self.debug: config.general.exp_name = 'Test'
#         self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
#         pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
#         # print('pathlib.Path')
#         self.start_epoch = 0
#         self.only_infer = config.general.only_infer
#         self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1
#         # print('self.num_epochs:',self.num_epochs)
#         # data 读入训练和验证数据，数据是numpy格式
#         # train_ds = CINE2DT(config=config.data, mode='train')
#         # 用于测试--val
#         train_ds = CINE2DT(config=config.data, mode='val')
#         print('train_ds')
#         test_ds = CINE2DT(config=config.data, mode='val')
#         print('test_ds')
#         # self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
#         #                                pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
#         # print('train_loader')
#         # self.test_loader = DataLoader(dataset=test_ds, num_workers=0, drop_last=False, batch_size=1, shuffle=False)
#         # print('test_loader')
#         # network
#         # self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
#         # self.network.initialize_weights()
#         # self.network.cuda()
#         self.local_rank = local_rank
#         self.world_size = torch.cuda.device_count()
        
#         # 初始化分布式环境
#         # torch.distributed.init_process_group("nccl", rank=local_rank, world_size=torch.cuda.device_count())

#         # 定义 self.train_dataset 变量
#         # self.train_dataset = CINE2DT(config=config.data, mode='train')

#         # 使用 DistributedSampler 创建 DataLoader 对象
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
#         test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
#         self.train_loader = torch.utils.data.DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,batch_size=config.training.batch_size, sampler=train_sampler)
#         self.test_loader = DataLoader(dataset=test_ds, num_workers=0, drop_last=False, batch_size=1, sampler=test_sampler)

#         # 重新定义模型之前
#         torch.cuda.empty_cache()
#         # 使用 DistributedDataParallel
#         # network
#         self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
#         self.network = self.network.to(local_rank)
#         self.network.initialize_weights()
#         self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[local_rank],find_unused_parameters=True)
#         print("Parameter Count: %d" % count_parameters(self.network))

#         # optimizer
#         param_groups = add_weight_decay(self.network, config.training.optim_weight_decay)
#         self.optimizer = eval(f'torch.optim.{config.optimizer.which}')(param_groups, **eval(f'config.optimizer.{config.optimizer.which}').__dict__)
#         # 判断配置（config）中的 training.restore_ckpt 属性是否为 True。
#         # 如果是 True，表示希望从之前保存的检查点恢复模型，那么就会调用 self.load_model 方法，
#         # 并传入 config.training 作为参数，启动恢复模型的相关操作。
#         if config.training.restore_training: self.load_model(config.training)
#         self.loss_scaler = NativeScaler()

#     # def load_model(self, args):

#     #     if os.path.isdir(args.restore_ckpt):
#     #         args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
#     #     ckpt = torch.load(args.restore_ckpt)
#     #     self.network.load_state_dict(ckpt['model'], strict=True)

#     #     print("Resume checkpoint %s" % args.restore_ckpt)
#     #     if args.restore_training:
#     #         self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#     #         self.start_epoch = ckpt['epoch'] + 1
#     #         # self.loss_scaler.load_state_dict(ckpt['scaler'])
#     #         print("With optim & sched!")
#     def load_model(self, args):
#         if os.path.isdir(args.restore_ckpt):
#             # args.restore_ckpt = max(glob.glob(f'{args.resture_ckpt}/*.pth'), key=os.path.getmtime)
#             args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
#         ckpt = torch.load(args.restore_ckpt)
#         self.network.load_state_dict(ckpt['model'], strict=True)
#         self.start_epoch = ckpt['epoch'] + 1
#         if 'optimizer_state_dict' in ckpt:
#             self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#         if 'scaler' in ckpt and hasattr(self, 'loss_scaler'):
#             self.loss_scaler.load_state_dict(ckpt['scaler'])
#         print("Resume checkpoint %s" % args.restore_ckpt)

#     def save_model(self, epoch):
#         ckpt = {'epoch': epoch,
#                 'model': self.network.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 # 'scaler': self.loss_scaler.state_dict()
#                 }
#         torch.save(ckpt, f'{self.experiment_dir}/model_{epoch+1:03d}.pth')


# class TrainerKInterpolator(TrainerAbstract):

#     def __init__(self, config,local_rank):
#         print("TrainerKInterpolator initialized.")
#         super().__init__(config=config,local_rank=local_rank)
#         # 其他初始化代码
#         # print("TrainerKInterpolator initialized.")
#         self.train_criterion = CriterionKGIN(config.train_loss)
#         self.eval_criterion = CriterionKGIN(config.eval_loss)
#         self.logger = Logger()
#         self.scheduler_info = config.scheduler
        
#         # self.local_rank = local_rank
#         # self.world_size = torch.cuda.device_count()
        
#         # # 初始化分布式环境
#         # torch.distributed.init_process_group("nccl", rank=local_rank, world_size=torch.cuda.device_count())

#         # # 定义 self.train_dataset 变量
#         # # self.train_dataset = CINE2DT(config=config.data, mode='train')

#         # # 使用 DistributedSampler 创建 DataLoader 对象
#         # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds)
#         # self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=config.train.batch_size, sampler=train_sampler)
#         # self.test_loader = DataLoader(dataset=self.test_ds, num_workers=0, drop_last=False, batch_size=1, sampler=test_sampler)

#         # 使用 DistributedDataParallel
#           # 使用 DistributedDataParallel
#         # self.network = self.network.to(local_rank)
#         # self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[local_rank])


#         # self.config = config
#         # self.local_rank = local_rank
#         self.device = torch.device(f'cuda:{self.local_rank}')
#         # self.network = self.build_model().to(self.device)
#         # self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank])

#         # # 修改数据加载器以支持分布式
#         # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
#         # self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=config.train.batch_size, sampler=train_sampler)
#         # 使用 train_ds 变量创建 DataLoader 对象
#         # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds)
#         # self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=config.train.batch_size, sampler=train_sampler)

#         # # 使用 DistributedDataParallel
#         # self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank])
       

#     def run(self):
#         print("Starting run method")
#         # 数据加载
#         # print("Loading data")
#         # 模型初始化
#         # print("Initializing model")
#         # 训练循环
#         # print("Starting training loop")
#         # 初始化tqdm进度条
#         # pbar = tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=100)
#         # start_time = time.time()

#         pbar = tqdm.tqdm(range(self.start_epoch, self.num_epochs))
#         for epoch in pbar:
#             self.logger.reset_metric_item()
#             start_time = time.time()
#             print('run-start_time:',start_time)
#             if not self.only_infer:
#                 self.train_one_epoch(epoch)
#             self.run_test()
#             self.logger.update_metric_item('train/epoch_runtime', (time.time() - start_time)/60)
#             # if epoch % self.config.weights_save_frequency == 0 and not self.debug and epoch > 100:
#             if epoch % self.config.weights_save_frequency == 0:
#                 self.save_model(epoch)
#             if epoch == self.num_epochs - 1:
#                 self.save_model(epoch)
#             if not self.debug:
#                 self.logger.wandb_log(epoch)
#         print("Training completed")

#     def train_one_epoch(self, epoch):
#         print('train_one_epoch')
#         start_time = time.time()
#         # 累计损失
#         running_loss = 0.0
#         # 将模型切换到训练模式，启用 dropout 和 batch normalization 等训练特性。
#         self.network.train()
#         # print('self.network.train')
#         self.train_loader.sampler.set_epoch(epoch)  # 更新 DistributedSampler 的 epoch
#         # 通过for循环遍历 self.train_loader，从数据加载器中逐批获取 kspace（k空间数据）、coilmaps（线圈映射数据）和 sampling_mask（采样掩码）。
#         for i, (kspace, coilmaps, sampling_mask) in enumerate(self.train_loader):
#             kspace,coilmaps,sampling_mask = kspace.to(self.device), coilmaps.to(self.device), sampling_mask.to(self.device)
#             # 将多线圈的 k 空间数据和线圈映射数据转换成单一的k空间和图像域数据。
#             # ref_kspace 是处理后的单线圈 k 空间数据，ref_img 是相应的图像域数据。
#             ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
#             # train_one_epoch-ref_kspace: torch.Size([16, 18, 192, 192])
#             # train_one_epoch-ref_img: torch.Size([16, 18, 192, 192])
#             # train_one_epoch-kspace: torch.Size([16, 18, 192, 192])
#             # print('train_one_epoch-ref_kspace:',ref_kspace.shape)
#             # print('train_one_epoch-ref_img:',ref_img.shape)
#             # torch.unsqueeze(sampling_mask, dim=2)是在第二个维度（dim=2）上增加一个维度，
#             # 使得sampling_mask可以与ref_kspace进行广播乘法。这样做的目的可能是为了应用某种采样模式或者对k空间数据进行加权。
#             # kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2) #[1,18,1,192]
#             kspace = ref_kspace
#             # print('train_one_epoch-kspace:',kspace.shape)

#             self.optimizer.zero_grad()
#             adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

#             # with torch.cuda.amp.autocast(enabled=False):
#             with torch.amp.autocast('cuda', enabled=False):
#                 # pred_list, im_recon  # 返回所有调整阶段的预测结果列表和重建的图像
#                 k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
#                 sampling_mask = sampling_mask.repeat_interleave(ref_kspace.shape[2], 2)
#                 ls = self.train_criterion(k_recon_2ch, torch.view_as_real(ref_kspace), im_recon, ref_img, kspace_mask=sampling_mask)

#                 self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())
#             # 使用 reduce 将每个进程的损失值聚合到主进程
#             loss_reduced = reduce_tensor(ls['k_recon_loss_combined'], self.world_size)
            
#             running_loss += loss_reduced.item()
#             # 添加打印信息
#             current_lr = self.optimizer.param_groups[0]['lr']
#             elapsed_time = time.time() - start_time
#             eta = datetime.timedelta(seconds=int((elapsed_time / (i + 1)) * (len(self.train_loader) - (i + 1))))
#             max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

#             # 更新tqdm显示信息
#             # pbar.set_description(
#             #     f"Epoch: [{epoch}] [{i + 1}/{len(self.train_loader)}] eta: {str(eta)} "
#             #     f"lr: {current_lr:.6f} loss: {loss_reduced.item():.4f} ({running_loss / (i + 1):.4f}) "
#             #     f"time: {elapsed_time / (i + 1):.4f} data: 0.0002 max mem: {max_memory:.0f}"
#             # )
#             # Log the detailed information
#             print(
#                 f"Epoch: [{epoch}] [{i + 1}/{len(self.train_loader)}] eta: {str(eta)} "
#                 f"lr: {current_lr:.6f} loss: {loss_reduced.item():.4f} ({running_loss / (i + 1):.4f}) "
#                 f"time: {elapsed_time / (i + 1):.4f} data: 0.0002 max mem: {max_memory:.0f}"
#             )
            
#             torch.cuda.empty_cache()
#             self.logger.update_metric_item('train/k_recon_loss', ls['k_recon_loss'].item()/len(self.train_loader))
#             self.logger.update_metric_item('train/recon_loss', ls['photometric'].item()/len(self.train_loader))

#     def run_test(self):
#         # 初始化 out，形状为 [118, 18, 192, 192] 的复数张量，用于存储所有批次（118 个）的预测 k 空间数据
#         out = torch.complex(torch.zeros([118, 18, 192, 192]), torch.zeros([118, 18, 192, 192])).to(self.device)
#         # 切换模型到评估模式 (eval)，禁用 dropout 和 batch normalization 的动态行为。
#         self.network.eval()
#         # 使用 barrier 同步所有进程
#         torch.distributed.barrier()
#         # 禁用梯度计算，节省内存并提高推理速度
#         with torch.no_grad():
#             for i, (kspace, coilmaps, sampling_mask) in enumerate(self.test_loader):
#                 # coilmaps：接收线圈的感应场（即 coil sensitivity maps）。
#                 kspace,coilmaps,sampling_mask = kspace.to(self.device), coilmaps.to(self.device), sampling_mask.to(self.device)
#                 ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
                
#                 # wandb.log({"ref_kspace": ref_kspace})
#                 # wandb.log({"ref_kspace_abs": torch.abs(ref_kspace), "ref_kspace_angle": torch.angle(ref_kspace)})
#                 # wandb.log({"ref_kspace": ref_kspace}, json_encoder=MyEncoder)
#                 # wandb.log({"ref_kspace-shape": ref_kspace.shape})
#                 # wandb.log({"coilmaps": coilmaps}, json_encoder=MyEncoder)
#                 # wandb.log({"coilmaps :":coilmaps})
#                 # wandb.log({"ref_kspace": ref_kspace}, json_encoder=MyEncoder)
#                 # wandb.log({"coilmaps shape:":coilmaps.shape})
#                 # wandb.log({"sampling_mask:":sampling_mask})
#                 # wandb.log({"sampling_mask": sampling_mask}, json_encoder=MyEncoder)
#                 # wandb.log({"sampling_mask shape:":sampling_mask.shape})
                
#                 # 将参考的 k 空间数据 ref_kspace 与采样掩膜 sampling_mask 相乘，模拟部分采样。
#                 # kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2)
#                 kspace = ref_kspace
#                 # print('train_one_epoch-run_test-kspace:',kspace.shape)
#                 # pred_list, im_recon 返回所有调整阶段的预测结果列表和重建的图像
#                 k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask) # size of kspace and mask: [B, T, H, W]
#                 # k_recon_2ch：重建的k空间（可能是一个时间序列，取最后一帧）
#                 k_recon_2ch = k_recon_2ch[-1]

#                 kspace_complex = torch.view_as_complex(k_recon_2ch)
#                 # sampling_mask.repeat_interleave 在最后一个维度重复采样掩膜，用于处理 2D k 空间。
#                 sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)
                
#                 out[i] = kspace_complex

#                 ls = self.eval_criterion([kspace_complex], ref_kspace, im_recon, ref_img, kspace_mask=sampling_mask, mode='test')

#                 self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
#                 self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
#                 self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))
#             print('...', out.shape, out.dtype)
#             out = out.cpu().data.numpy()
#             np.save('out_1201.npy', out)
#             print('save success......')
            
#             # 使用 reduce 将每个进程的损失值聚合到主进程
#             loss_reduced = reduce_tensor(ls['k_recon_loss'], self.world_size)
#             psnr_reduced = reduce_tensor(ls['psnr'], self.world_size)
#             self.logger.update_metric_item('val/k_recon_loss', loss_reduced.item()/len(self.test_loader))
#             self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
#             self.logger.update_metric_item('val/psnr', psnr_reduced.item()/len(self.test_loader))

#             self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
#             self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr'])

# def reduce_tensor(tensor, world_size):
#     rt = tensor.clone()
#     torch.distributed.reduce(rt, dst=0, op=torch.distributed.ReduceOp.SUM)
#     rt /= world_size
#     return rt


# # wandb.log({"ref_kspace": ref_kspace}, json_encoder=MyEncoder)


#     # def train_one_epoch(self, epoch):
#     #     self.network.train()
#     #     for i, batch in enumerate(self.train_loader):
#     #         kspace, sampling_mask = [item.cuda() for item in batch[0]][:]
#     #         ref = batch[1][0].cuda()

#     #         self.optimizer.zero_grad()
#     #         adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

#     #         with torch.cuda.amp.autocast(enabled=False):
#     #             k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
#     #             sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)
#     #             ls = self.train_criterion(k_recon_2ch, torch.view_as_real(kspace), im_recon, ref, kspace_mask=sampling_mask)

#     #             self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())

#     #         self.logger.update_metric_item('train/k_recon_loss', ls['k_recon_loss'].item()/len(self.train_loader))
#     #         self.logger.update_metric_item('train/recon_loss', ls['photometric'].item()/len(self.train_loader))

#     # def run_test(self):
#     #     self.network.eval()
#     #     with torch.no_grad():
#     #         for i, batch in enumerate(self.test_loader):
#     #             kspace, sampling_mask = [item.cuda() for item in batch[0]][:]
#     #             ref = batch[1][0].cuda()

#     #             k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask) # size of kspace and mask: [B, T, H, W]
#     #             k_recon_2ch = k_recon_2ch[-1]

#     #             kspace_complex = torch.view_as_complex(k_recon_2ch)
#     #             sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)

#     #             ls = self.eval_criterion([kspace_complex], kspace, im_recon, ref, kspace_mask=sampling_mask, mode='test')

#     #             self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
#     #             self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
#     #             self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))

#     #         self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
#     #         self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr'])
