import os
import sys
import pathlib
import torch
import glob
import tqdm
import time
import sys 
sys.path.append('/nfs/zzy/code/k_gin_base')  
from torch.utils.data import DataLoader
from dataset.dataloader import CINE2DT
# from model.k_interpolator import KInterpolator
from varnet import VarNet
from losses import CriterionKGIN

from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, NativeScalerWithGradNormCount as NativeScaler, add_weight_decay
from utils import multicoil2single
import numpy as np
import datetime

# PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
os.environ['OMP_NUM_THREADS'] = '1'
# 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 指定使用 GPU 1 和 GPU 4
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4

# 设置环境变量 CUDA_VISIBLE_DEVICES  0-5(nvidia--os) 2-6 3-7
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'  # 指定使用 GPU 7 和 GPU 3
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 指定使用 GPU 4 和 GPU 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainerAbstract:
    def __init__(self, config):
        print("TrainerAbstract initialized.")
        super().__init__()
        self.config = config.general
        self.debug = config.general.debug
        if self.debug: config.general.exp_name = 'test_varnet'
        self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.only_infer = config.general.only_infer
        self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1

        # data
        # train_ds = CINE2DT(config=config.data, mode='train')
        train_ds = CINE2DT(config=config.data, mode='val')
        test_ds = CINE2DT(config=config.data, mode='val')
        self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
                                       pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_ds, num_workers=2, drop_last=False, batch_size=1, shuffle=False)

        # network
        self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
        print(type(self.network))
        print(self.network.__dict__)
        # self.network.initialize_weights()
        self.network.cuda()
        # Parameter Count: 29452068
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
            #train_one_epoch-kspace torch.Size([2, 20, 18, 192, 192])
            print('train_one_epoch-kspace', kspace.shape)
            # train_one_epoch-coilmaps torch.Size([4, 20, 1, 192, 192])
            # train_one_epoch-coilmaps torch.Size([2, 20, 1, 192, 192])
            print('train_one_epoch-coilmaps', coilmaps.shape)
            # train_one_epoch-sampling_mask torch.Size([4, 18, 192])
            # train_one_epoch-sampling_mask torch.Size([2, 18, 192])
            print('train_one_epoch-sampling_mask', sampling_mask.shape)
            # 确保 mask 是布尔类型
            input_mask = sampling_mask.to(torch.bool)  
            ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
            # train_one_epoch-ref_kspace torch.Size([4, 18, 192, 192])
            # train_one_epoch-ref_kspace torch.Size([2, 18, 192, 192])
            print('train_one_epoch-ref_kspace', ref_kspace.shape)
            # train_one_epoch-ref_img torch.Size([4, 18, 192, 192])
            # train_one_epoch-ref_img torch.Size([2, 18, 192, 192])
            print('train_one_epoch-ref_img', ref_img.shape)
            # kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2) #[1,18,1,192]
            kspace = ref_kspace

            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            with torch.cuda.amp.autocast(enabled=False):
                # k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
                k_recon_2ch, im_recon = self.network(kspace, mask=input_mask)  # size of kspace and mask: [B, T, H, W]
                # AttributeError: 'list' object has no attribute 'shape'
                # print('train_one_epoch-k_recon_2ch', k_recon_2ch.shape)
                # train_one_epoch-im_recon torch.Size([4, 18, 192, 192])
                print('train_one_epoch-im_recon', im_recon.shape)
                sampling_mask = sampling_mask.repeat_interleave(ref_kspace.shape[2], 2)
                # train_one_epoch-sampling_mask-2 torch.Size([4, 18, 36864])
                print('train_one_epoch-sampling_mask-2', sampling_mask.shape)
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
            # pbar.set_description(
            #     f"Epoch: [{epoch}] [{i + 1}/{len(self.train_loader)}] eta: {str(eta)} "
            #     f"lr: {current_lr:.6f} loss: {loss_reduced.item():.4f} ({running_loss / (i + 1):.4f}) "
            #     f"time: {elapsed_time / (i + 1):.4f} data: 0.0002 max mem: {max_memory:.0f}"
            # )
            # Log the detailed information
            if i % 20 ==0:
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
        self.network.eval()
        with torch.no_grad():
            for i, (kspace, coilmaps, sampling_mask) in enumerate(self.test_loader):
                kspace,coilmaps,sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
                ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
                # kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2)
                
                # np.save('out_1130_2.npy', out)
                kspace = ref_kspace

                k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask) # size of kspace and mask: [B, T, H, W]
                k_recon_2ch = k_recon_2ch[-1]

                kspace_complex = torch.view_as_complex(k_recon_2ch)
                sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)
                
                out[i] = kspace_complex

                ls = self.eval_criterion([kspace_complex], ref_kspace, im_recon, ref_img, kspace_mask=sampling_mask, mode='test')

                self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))
            print('...', out.shape, out.dtype)
            out = out.cpu().data.numpy()
            # np.save('out.npy', out)
            # np.save('out_1120.npy', out)
            # np.save('out_1130_3.npy', out)
            np.save('out_varnet_1212_test_1.npy', out)
            self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
            self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr'])