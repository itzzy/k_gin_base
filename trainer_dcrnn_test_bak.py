import os
import sys
import pathlib
import torch
import glob
import tqdm
import time
from torch.utils.data import DataLoader
from dataset.dataloader import CINE2DT
# from model.k_interpolator import KInterpolator
from model.model_pytorch import CRNN_MRI
import matplotlib.pyplot as plt

from os.path import join

from losses import CriterionKGIN
from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, NativeScalerWithGradNormCount as NativeScaler, add_weight_decay
from utils import multicoil2single
from utils.fastmriBaseUtils import FFT2c,fft2c_2d,fft2c
from utils import compressed_sensing as cs
from utils.dnn_io import to_tensor_format
from utils.dnn_io import from_tensor_format
from torch.autograd import Variable
from utils.metric import complex_psnr

from torch.utils.tensorboard import SummaryWriter


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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 指定使用 GPU 1 和 GPU 4

# 设置环境变量 CUDA_VISIBLE_DEVICES  0-5(nvidia--os) 2-6 3-7
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'  # 指定使用 GPU 7 和 GPU 3
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 指定使用 GPU 4 和 GPU 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
criterion = torch.nn.MSELoss()


class TrainerAbstract:
    def __init__(self, config):
        print("TrainerAbstract initialized.")
        super().__init__()
        self.config = config.general
        self.debug = config.general.debug
        if self.debug: config.general.exp_name = 'test_dcrnn_test'
        self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.only_infer = config.general.only_infer
        self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1
        # acc 加速倍速
        self.acc_rate = config.general.acc_rate
        # 通过索引取出列表中的元素（因为这里只有一个元素）并转换为整数类型
        self.acc_rate_value = int(self.acc_rate[0])
        print('acc_rate_value-type:',type(self.acc_rate_value))
        # print('self.acc_rate-dtype:',self.acc_rate.dtype)
        

        # data
        # train_ds = CINE2DT(config=config.data, mode='train')
        train_ds = CINE2DT(config=config.data, mode='val')
        test_ds = CINE2DT(config=config.data, mode='val')
        self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
                                       pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_ds, num_workers=2, drop_last=False, batch_size=1, shuffle=False)

        # network
        self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
        # self.network = getattr(sys.modules[__name__])(eval('config.network'))
        # self.network.initialize_weights()
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
        # Initialize SummaryWriter
        self.writer = SummaryWriter(log_dir=config.general.tensorboard_log_dir)

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
            self.optimizer.zero_grad()  # 清除梯度
            self.logger.reset_metric_item()
            start_time = time.time()
            if not self.only_infer:
                self.train_one_epoch(epoch)
            self.run_test()
            self.logger.update_metric_item('train/epoch_runtime', (time.time() - start_time)/60)
            print(f"Epoch {epoch+1}/{self.num_epochs} 完成，耗时：{(time.time() - start_time)/60:.2f} 分钟")
            # if epoch % self.config.weights_save_frequency == 0 and not self.debug and epoch > 150:
            if epoch % self.config.weights_save_frequency == 0:
                self.save_model(epoch)
            if epoch == self.num_epochs - 1:
                self.save_model(epoch)
            if not self.debug:
                self.logger.wandb_log(epoch)
        self.writer.close()

    def train_one_epoch(self, epoch):
        # start_time = time.time()
        # # 累计损失
        # running_loss = 0.0
        # self.network.train()
        # train_err = 0
        # train_batches = 0
        start_time = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        self.network.train()
        train_err = 0
        train_batches = 0
        for i, (kspace, coilmaps, sampling_mask) in enumerate(self.train_loader):
            kspace,coilmaps,sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
            # train_one_epoch-kspace torch.Size([4, 20, 18, 192, 192])
            # 18对应t 20-线圈数 4-slice
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
            
            # ref_img_real = c2r(ref_img)
            # print('train_one_epoch-ref_img_real', ref_img_real.shape)
            if ref_img.is_cuda:  # 判断张量是否在GPU上
                ref_img = ref_img.cpu()  # 如果在GPU上，将其复制到CPU上
            # train_one_epoch-kspace_real-dtype: torch.float32
            # print('train_one_epoch-ref_img_real-dtype:', ref_img_real.dtype)
            # train_one_epoch-ref_img torch.Size([2, 18, 192, 192])
            # print('train_one_epoch-ref_img', ref_img.shape)
            
            # 得到的mask: (2, 192, 192, 18)
            im_und, k_und, mask, im_gnd = prep_input(ref_img, self.acc_rate_value)
            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))
            '''
            train_one_epoch-im_u torch.Size([2, 2, 192, 192, 18])
            train_one_epoch-k_u torch.Size([2, 2, 192, 192, 18])
            train_one_epoch-mask torch.Size([2, 2, 192, 192, 18])
            '''
            # print('train_one_epoch-im_u', im_u.shape)
            # print('train_one_epoch-k_u', k_u.shape)
            # print('train_one_epoch-mask', mask.shape)
            
            
            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            with torch.cuda.amp.autocast(enabled=False):
                
                # rec = rec_net(im_u, k_u, mask, test=False)
                # k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
                # im_recon = self.network(ref_img_real, kspace_real,sampling_mask,test=False)  # size of kspace and mask: [B, T, H, W]
                im_recon = self.network(im_u, k_u,mask,test=False)  # size of kspace and mask: [B, T, H, W]
                # AttributeError: 'list' object has no attribute 'shape'
                # print('train_one_epoch-k_recon_2ch', k_recon_2ch.shape)
                # train_one_epoch-im_recon torch.Size([4, 18, 192, 192])
                
                # train_one_epoch-im_recon torch.Size([2, 2, 192, 192, 18])
                # train_one_epoch-im_recon-dtype: torch.float32
                # print('train_one_epoch-im_recon', im_recon.shape)
                # print('train_one_epoch-im_recon-dtype:', im_recon.dtype)
                k_recon_2ch = fft2c(im_recon)
                # 确保 im_recon 和 k_recon_2ch 是实数张量
                # if im_recon.is_complex():
                #     im_recon = torch.view_as_real(im_recon)  # 将复数张量转换为实数张量
                # if k_recon_2ch.is_complex():
                #     k_recon_2ch = torch.view_as_real(k_recon_2ch)  # 将复数张量转换为实数张量

                im_recon_4d = r2c_5d_to_4d(im_recon)
                # train_one_epoch-im_recon_4d torch.Size([2, 192, 192, 18])
                # train_one_epoch-im_recon_4d-dtype: torch.complex64
                # print('train_one_epoch-im_recon_4d', im_recon_4d.shape)
                # print('train_one_epoch-im_recon_4d-dtype:', im_recon_4d.dtype)
                k_recon_2ch_4d = r2c_5d_to_4d(k_recon_2ch)
                # print('train_one_epoch-k_recon_2ch')
                # k_recon_2ch_2 = fft2c_2d(im_recon)
                # print('train_one_epoch-k_recon_2ch-2')
                # im_recon_4d = r2c_5d_to_4d(im_recon)
                # print('train_one_epoch-im_recon_4d', im_recon_4d.shape)
                # print('train_one_epoch-im_recon_4d-dtype:', im_recon_4d.dtype)
                # k_recon_2ch_4d = r2c_5d_to_4d(k_recon_2ch)

                # print('train_one_epoch-k_recon_2ch_4d', k_recon_2ch_4d.shape)
                # print('train_one_epoch-k_recon_2ch_4d-dtype:', k_recon_2ch_4d.dtype)
                
                # loss = criterion(im_recon, gnd)
                # loss.backward()
                # self.optimizer.step()

            #     train_err += loss.item()
            #     train_batches += 1
            #     running_loss += loss.item()
            # epoch_loss = running_loss / train_batches if train_batches > 0 else 0
            # # 判断当前 epoch 是否是 10 的倍数，如果是则打印平均损失
            # if i % 10 == 0:
            #     print(f'Epoch {i} - Average Training Loss: {epoch_loss}')
                
                sampling_mask = sampling_mask.repeat_interleave(ref_kspace.shape[2], 2)
                # train_one_epoch-sampling_mask-2 torch.Size([4, 18, 36864])
                # print('train_one_epoch-sampling_mask-2', sampling_mask.shape)
                # ls = self.train_criterion(k_recon_2ch, torch.view_as_real(ref_kspace), im_recon, ref_img, kspace_mask=sampling_mask)
                # ls = self.train_criterion(k_recon_2ch_4d, torch.view_as_real(ref_kspace), im_recon_4d, ref_img, kspace_mask=sampling_mask)
                ls = self.train_criterion(k_recon_2ch_4d, ref_kspace, im_recon_4d, ref_img, kspace_mask=sampling_mask)
                # print('train_one_epoch-ls')
                # self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())
                # self.loss_scaler._scaler(ls['k_recon_loss_combined']).backward(retain_graph=True)
                # self.loss_scaler(None, self.optimizer, parameters=self.network.parameters())
                # self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters(), retain_graph=True)
                print('train_one_epoch-k_recon_loss_combined:',ls['k_recon_loss_combined'])
                self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())
                
                self.loss_scaler.update()  # 确保调用 update 方法
                
            # 检查梯度是否溢出
            if self.loss_scaler._has_overflow(self.optimizer):
                print("Gradient overflow detected!")
                self.loss_scaler.update_scale_(self.optimizer)
                self.loss_scaler._maybe_opt_step(self.optimizer)
                continue
             # 使用 reduce 将每个进程的损失值聚合到主进程
            loss_reduced = ls['k_recon_loss_combined']
            
            running_loss += loss_reduced.item()
            
            # Record loss to TensorBoard
            global_step = epoch * len(self.train_loader) + i
            self.writer.add_scalar('Loss/Train', loss_reduced.item(), global_step)

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
        model_name = 'test_dcrnn_test'
        # Configure directory info
        project_root = '.'
        # self.save_dir = join(project_root, 'models/%s' % model_name)
        self.save_dir = '/nfs/zzy/code/k_gin_base/tmp'  # 本地临时目录
        if not os.path.isdir(self.save_dir):
            print(f"Directory {self.save_dir} does not exist. Creating it...")
            os.makedirs(self.save_dir)
        # 初始化变量
        out = torch.complex(torch.zeros([118, 18, 192, 192]), torch.zeros([118, 18, 192, 192])).to(device)
        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        running_test_loss = 0.0
        epoch_test_loss = 0.0
        im_recon_list = []  # 用于存储 im_recon 张量


        self.network.eval()
        with torch.no_grad():
            for i, (kspace, coilmaps, sampling_mask) in enumerate(self.test_loader):
                kspace, coilmaps, sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
                
                # 将多通道 k-space 和图像转换为单通道
                ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
                kspace = ref_kspace

                # 如果图像在 GPU 上，将其转换到 CPU
                if ref_img.is_cuda:
                    ref_img = ref_img.cpu()

                # 准备输入数据 return im_und_l, k_und_l, mask_l, im_gnd_l
                im_und, k_und, mask, im_gnd = prep_input(ref_img, self.acc_rate_value)
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

                # 网络预测
                im_recon = self.network(im_u, k_u, mask, test=False)
                # print('run_test-im_recon-shape:',im_recon.shape)
                # print('run_test-im_recon-dtype:',im_recon.dtype)
                
                
                im_recon_list.append(im_recon.cpu().data.numpy())  # 将 im_recon 转换为 numpy 数组并添加到列表中

                k_recon_2ch = fft2c(im_recon)
                im_recon_4d = r2c_5d_to_4d(im_recon)
                k_recon_2ch_4d = r2c_5d_to_4d(k_recon_2ch)
                sampling_mask = sampling_mask.repeat_interleave(ref_kspace.shape[2], 2)
                ls = self.eval_criterion(k_recon_2ch_4d, ref_kspace, im_recon_4d, ref_img, kspace_mask=sampling_mask)
                # ls = self.eval_criterion([kspace_complex], ref_kspace, im_recon, ref_img, kspace_mask=sampling_mask, mode='test')

                self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))
                
                
                # 计算损失
                # loss = criterion(im_recon, gnd)
                # running_test_loss += loss.item()

                # 计算 PSNR
                for im_i, und_i, pred_i in zip(
                    from_tensor_format(im_gnd.numpy()),
                    from_tensor_format(im_und.numpy()),
                    from_tensor_format(im_recon.data.cpu().numpy())
                ):
                    base_psnr += complex_psnr(im_i, und_i, peak='max')
                    test_psnr += complex_psnr(im_i, pred_i, peak='max')

                # 保存重建图像
                if i % 10 == 0:
                    vis.append((
                        from_tensor_format(im_gnd.numpy())[0],
                        from_tensor_format(im_recon.data.cpu().numpy())[0],
                        from_tensor_format(im_und.numpy())[0],
                        from_tensor_format(mask.data.cpu().numpy(), mask=True)[0]
                    ))

                test_batches += 1

                # 打印中间测试结果
                # if i % 10 == 0:
                #     epoch_test_loss = running_test_loss / test_batches if test_batches > 0 else 0
                #     print(f"Batch {i} - Average Test Loss: {epoch_test_loss:.6f}")

            # 计算最终平均损失和 PSNR
            # epoch_test_loss = running_test_loss / test_batches if test_batches > 0 else 0
            base_psnr /= (test_batches * self.test_loader.batch_size)
            test_psnr /= (test_batches * self.test_loader.batch_size)

            # print(f"Final Test Loss: {epoch_test_loss:.6f}")
            print(f"Base PSNR: {base_psnr:.6f}")
            print(f"Test PSNR: {test_psnr:.6f}")
            
            # # 将 im_recon 保存为.npy 文件
            # im_recon_array = np.concatenate(im_recon_list, axis=0)  # 拼接所有 im_recon 张量
            # np.save(join(self.save_dir, 'im_recon.npy'), im_recon_array)  # 保存为.npy 文件
             # 将 im_recon 保存为.npy 文件
            im_recon_array = np.concatenate(im_recon_list, axis=0)  # 拼接所有 im_recon 张量
            # print(im_recon_array.nbytes / (1024 ** 3), "GB")
            # print(im_recon_array.dtype)
            print("Save directory:", self.save_dir)
            print("Data shape:", im_recon_array.shape)
            print("Data dtype:", im_recon_array.dtype)
            print("Data size:", im_recon_array.nbytes / (1024 ** 3), "GB")
            np.save(join(self.save_dir, 'im_recon.npy'), im_recon_array)  # 保存为.npy 文件

            # 从 im_recon 生成 k-space 数据并保存
            # kspace_recon = torch.fft.fft2(torch.tensor(im_recon_array))  # 对重建的图像进行傅里叶变换得到 k-space 数据
            # kspace_recon = torch.view_as_complex(kspace_recon)  # 转换为复数形式
            # np.save(join(self.save_dir, 'kspace_recon.npy'), kspace_recon.cpu().numpy())  # 保存为 .npy 文件
            # 从 im_recon 生成 k-space 数据并保存
            kspace_recon = torch.fft.fft2(torch.tensor(im_recon_array))  # 对重建的图像进行傅里叶变换得到 k-space 数据
            np.save(join(self.save_dir, 'kspace_recon.npy'), kspace_recon.cpu().numpy())  # 直接保存复数类型的 k-space 数据


        # 保存图像和模型
        i = 0
        for im_i, pred_i, und_i, mask_i in vis:
            im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
            if i%50 == 0:
                plt.imsave(join(self.save_dir, f'im_{i}_x.png'), im, cmap='gray')

            im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                                    im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
            if i%50 == 0:
                plt.imsave(join(self.save_dir, f'im_{i}_t.png'), im, cmap='gray')
                plt.imsave(join(self.save_dir, f'mask_{i}.png'),
                    np.fft.fftshift(mask_i[..., 0]), cmap='gray')
            i += 1

        # 保存网络权重
        model_path = join(self.save_dir, "final_model.pth")
        torch.save(self.network.state_dict(), model_path)
        print(f"Model parameters saved at {model_path}")
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
    # c2r-kspace_real_imag-shape-1: torch.Size([4, 18, 192, 192, 2])
    # print('c2r-kspace_real_imag-shape-1:',kspace_real_imag.shape)

    # 调整维度顺序，将表示实部和虚部的维度放到第二个维度，同时把time_steps维度调整到最后
    # 转换后的形状变为 (batch_size, 2, height, width, time_steps)
    kspace_real_imag = kspace_real_imag.permute(0, 4, 2, 3, 1)
    # print('c2r-kspace_real_imag-shape-2:',kspace_real_imag.shape)

    return kspace_real_imag

def r2c_5d_to_4d(x):
    """
    将五维实数张量转换为四维复数张量。
    输入形状: [batch_size, 2, height, width, channels]
    输出形状: [batch_size, height, width, channels]
    """
    # 检查输入是否为numpy数组，如果是，则转换为Tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    # 假设x的形状为[batch_size, 2, height, width, channels]
    # 我们将第一个维度（大小为2）的实部和虚部分别取出
    re = x[:, 0, :, :, :]  # 实部
    im = x[:, 1, :, :, :]  # 虚部
    
    # 检查 re 和 im 是否为复数张量，如果是则提取实部和虚部
    if re.is_complex():
        re = re.real
    if im.is_complex():
        im = im.real
    
    # 创建复数张量
    x_complex = torch.complex(re, im)
    
    # 返回四维复数张量，形状为[batch_size, height, width, channels]
    return x_complex

# def r2c_5d_to_4d(x):
#     # 检查输入是否为numpy数组，如果是，则转换为Tensor
#     if isinstance(x, np.ndarray):
#         x = torch.from_numpy(x)
    
#     # 假设x的形状为[batch_size, 2, height, width, channels]
#     # 我们将第一个维度（大小为2）的实部和虚部分别取出
#     re = x[:, 0, :, :, :]  # 实部
#     im = x[:, 1, :, :, :]  # 虚部
    
#     # 创建复数张量
#     x_complex = torch.complex(re, im)
    
#     # 返回四维张量，形状为[batch_size, height, width, channels]
#     # 使用torch.view_as_real将复数张量转换为实数张量
#     output_real_tensor = torch.view_as_real(x_complex)
#     return output_real_tensor

def prep_input(im, acc=4.0):
    """
    Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                    higher the value, more undersampling
    """
    # 调整mask维度顺序使其符合后续操作要求（如果需要的话，根据实际情况调整）
    # 假设原本的操作期望mask维度顺序为 (batch_size, height, width)，而新生成的mask维度顺序不符合，进行如下调整
    # if len(mask.shape) == 2:  # 假设新生成的mask是二维的，若实际情况不同需相应修改判断条件
    #     mask = np.expand_dims(mask, axis=0)  # 添加batch_size维度（这里假设batch_size维度为0，根据实际调整）
    # mask = np.transpose(mask, (0, 2, 1))  # 调整height和width维度顺序，同样根据实际期望顺序调整
    
    # 扩展 mask 以匹配 ref_img 的维度 [batch, time, height, width]
    # train_one_epoch-ref_img torch.Size([2, 18, 192, 192])
    # print('prep_input-im-shape:', im.shape)
    batch_size, time, height, width = im.shape
    # mask = get_cine_mask(acc, x=width, y=height)  # x 和 y 要与输入图像的宽度和高度一致
    # mask = get_cine_mask(int(acc), x=width, y=height)
    mask = get_cine_mask(int(acc), x=time, y=height)
    '''
    prep_input-mask-shape: (192, 18)
    prep_input-mask-dtype: float64
    '''
    # print('prep_input-mask-shape:', mask.shape)
    # print('prep_input-mask-dtype:', mask.dtype)
    # 对 mask 进行转置操作  class CINE2DT(torch.utils.data.Dataset)有以下代码：
    # self.mask = np.transpose(self.mask,[1,0])
    mask = np.transpose(mask,[1,0])
    
    mask = np.expand_dims(mask, axis=0)  # 添加 batch 维度
    mask = np.expand_dims(mask, axis=0)  # 添加 time 维度
    # mask = np.tile(mask, (batch_size, time, 1, 1))  # 广播到完整形状
    # 得到的mask: (2, 192, 192, 18) (2, 192, 18, 192)
    mask = np.tile(mask, (batch_size, width, 1, 1))  # 广播到完整形状
    # AttributeError: 'numpy.ndarray' object has no attribute 'permute'
    # mask = mask.permute(0,3,2,1)
    # 将 NumPy 数组转换为 PyTorch 张量
    mask_tensor = torch.from_numpy(mask)

    # 使用 permute 方法重新排列维度
    # mask_permuted = mask_tensor.permute(0, 3, 2, 1)
    mask_permuted = mask_tensor.permute(0, 2, 1, 3)
    # prep_input-mask_permuted-shape: torch.Size([2, 18, 192, 192])
    print('prep_input-mask_permuted-shape:', mask_permuted.shape)
    
    # 将 mask 转为 torch.Tensor，并调整为网络接受的格式
    # mask_l = torch.from_numpy(mask).to(dtype=torch.float32)  # 转换数据类型为 float32
    # # prep_input-mask_l-shape: torch.Size([2, 18, 192, 18])
    # # prep_input-mask_l-dtype: torch.float32
    # print('prep_input-mask_l-shape:', mask_l.shape)
    # print('prep_input-mask_l-dtype:', mask_l.dtype)
    # mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
    # mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
    # prep_input-mask_l-shape-1: torch.Size([2, 2, 192, 18, 192])
    # prep_input-mask_l-dtype-1: torch.float64
    mask_l = mask_l.permute(0, 1, 2, 4, 3)
    # prep_input-mask_l-shape-1: torch.Size([1, 2, 192, 192, 18])
    # prep_input-mask_l-dtype-1: torch.float64
    # print('prep_input-mask_l-shape-1:',mask_l.shape)
    # print('prep_input-mask_l-dtype-1:',mask_l.dtype)
    # 使用 permute 方法重新排列维度
    # adjusted_mask = mask_l.permute(0, 1, 2, 4, 3)
    # mask_l = adjusted_mask
    # im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    # 对输入图像进行下采样
    # 将输入图像转换为 numpy 格式（如果 im 是 torch.Tensor）
    im_np = im.numpy() if isinstance(im, torch.Tensor) else im
    mask_np = mask_permuted.numpy() if isinstance(mask_permuted, torch.Tensor) else mask_permuted
    # prep_input-mask_np-shape: (2, 18, 192, 192)
    # prep_input-im_np-shape: (2, 18, 192, 192)
    # print('prep_input-mask_np-shape:', mask_np.shape)
    # print('prep_input-im_np-shape:', im_np.shape)
    # im_und, k_und = cs.undersample(im_np, mask, centred=False, norm='ortho')
    im_und, k_und = cs.undersample(im_np, mask_np, centred=False, norm='ortho')
    # im_und, k_und = cs.undersample(im_np, mask_np, centred=True, norm='ortho')
    # prep_input-im_und-shape: (1, 18, 192, 192)
    # print('prep_input-im_und-shape:', im_und.shape)
 
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    # prep_input-im_gnd_l-shape: torch.Size([1, 2, 192, 192, 18])
    # prep_input-im_und_l-shape: torch.Size([1, 2, 192, 192, 18])
    # prep_input-k_und_l-shape: torch.Size([1, 2, 192, 192, 18])
    # print('prep_input-im_gnd_l-shape:', im_gnd_l.shape)
    # print('prep_input-im_und_l-shape:', im_und_l.shape)
    # print('prep_input-k_und_l-shape:', k_und_l.shape)

    # 根据新mask的结构和维度，调整mask转换为张量的方式以及维度处理（示例，需根据实际调整）
    # mask_l = torch.from_numpy(mask.astype(np.float32))  # 转换数据类型为float32（假设符合后续要求，根据实际调整）
    # if len(mask_l.shape) == 3:  # 如果mask_l维度是3维，添加通道维度等操作（根据实际网络输入要求调整）
    #     mask_l = mask_l.unsqueeze(1)  # 在维度1的位置添加通道维度，假设符合网络对mask输入维度要求
    # print('prep_input-mask_l-shape-2:', mask_l.shape)
    # print('prep_input-mask_l-dtype:-2', mask_l.dtype)

    return im_und_l, k_und_l, mask_l, im_gnd_l
# def prep_input(im, acc=4.0):
#     """
#     Undersample the batch, then reformat them into what the network accepts.

#     Parameters
#     ----------
#     gauss_ivar: float - controls the undersampling rate.
#                     higher the value, more undersampling
#     """
#     # 调整mask维度顺序使其符合后续操作要求（如果需要的话，根据实际情况调整）
#     # 假设原本的操作期望mask维度顺序为 (batch_size, height, width)，而新生成的mask维度顺序不符合，进行如下调整
#     # if len(mask.shape) == 2:  # 假设新生成的mask是二维的，若实际情况不同需相应修改判断条件
#     #     mask = np.expand_dims(mask, axis=0)  # 添加batch_size维度（这里假设batch_size维度为0，根据实际调整）
#     # mask = np.transpose(mask, (0, 2, 1))  # 调整height和width维度顺序，同样根据实际期望顺序调整
    
#     # 扩展 mask 以匹配 ref_img 的维度 [batch, time, height, width]
#     #             # train_one_epoch-ref_img torch.Size([2, 18, 192, 192])
#     batch_size, time, height, width = im.shape
#     # mask = get_cine_mask(acc, x=width, y=height)  # x 和 y 要与输入图像的宽度和高度一致
#     # mask = get_cine_mask(int(acc), x=width, y=height)
#     mask = get_cine_mask(int(acc), x=time, y=height)
#     '''
#     prep_input-mask-shape: (192, 18)
#     prep_input-mask-dtype: float64
#     '''
#     print('prep_input-mask-shape:', mask.shape)
#     print('prep_input-mask-dtype:', mask.dtype)
    
#     mask = np.expand_dims(mask, axis=0)  # 添加 batch 维度
#     mask = np.expand_dims(mask, axis=0)  # 添加 time 维度
#     # mask = np.tile(mask, (batch_size, time, 1, 1))  # 广播到完整形状
#     # 得到的mask: (2, 192, 192, 18)
#     mask = np.tile(mask, (batch_size, width, 1, 1))  # 广播到完整形状
#     # AttributeError: 'numpy.ndarray' object has no attribute 'permute'
#     # mask = mask.permute(0,3,2,1)
#     # 将 NumPy 数组转换为 PyTorch 张量
#     mask_tensor = torch.from_numpy(mask)

#     # 使用 permute 方法重新排列维度
#     mask_permuted = mask_tensor.permute(0, 3, 2, 1)
#     # prep_input-mask_permuted-shape: torch.Size([2, 18, 192, 192])
#     print('prep_input-mask_permuted-shape:', mask_permuted.shape)
    
#     # 将 mask 转为 torch.Tensor，并调整为网络接受的格式
#     # mask_l = torch.from_numpy(mask).to(dtype=torch.float32)  # 转换数据类型为 float32
#     # # prep_input-mask_l-shape: torch.Size([2, 18, 192, 18])
#     # # prep_input-mask_l-dtype: torch.float32
#     # print('prep_input-mask_l-shape:', mask_l.shape)
#     # print('prep_input-mask_l-dtype:', mask_l.dtype)
#     # mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
#     # mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
#     mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
#     # prep_input-mask_l-shape-1: torch.Size([2, 2, 192, 18, 192])
#     # prep_input-mask_l-dtype-1: torch.float64
#     mask_l = mask_l.permute(0, 1, 2, 4, 3)
#     print('prep_input-mask_l-shape-1:',mask_l.shape)
#     print('prep_input-mask_l-dtype-1:',mask_l.dtype)
#     # 使用 permute 方法重新排列维度
#     # adjusted_mask = mask_l.permute(0, 1, 2, 4, 3)
#     # mask_l = adjusted_mask
#     # im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
#     # 对输入图像进行下采样
#     # 将输入图像转换为 numpy 格式（如果 im 是 torch.Tensor）
#     im_np = im.numpy() if isinstance(im, torch.Tensor) else im
#     mask_np = mask_permuted.numpy() if isinstance(mask_permuted, torch.Tensor) else mask_permuted
#     # prep_input-mask_np-shape: (2, 18, 192, 192)
#     # prep_input-im_np-shape: (2, 18, 192, 192)
#     print('prep_input-mask_np-shape:', mask_np.shape)
#     print('prep_input-im_np-shape:', im_np.shape)
#     # im_und, k_und = cs.undersample(im_np, mask, centred=False, norm='ortho')
#     im_und, k_und = cs.undersample(im_np, mask_np, centred=False, norm='ortho')
#     # im_und, k_und = cs.undersample(im_np, mask_np, centred=True, norm='ortho')
#     print('prep_input-im_und-shape:', im_und.shape)
 
#     im_gnd_l = torch.from_numpy(to_tensor_format(im))
#     im_und_l = torch.from_numpy(to_tensor_format(im_und))
#     k_und_l = torch.from_numpy(to_tensor_format(k_und))

#     # 根据新mask的结构和维度，调整mask转换为张量的方式以及维度处理（示例，需根据实际调整）
#     # mask_l = torch.from_numpy(mask.astype(np.float32))  # 转换数据类型为float32（假设符合后续要求，根据实际调整）
#     # if len(mask_l.shape) == 3:  # 如果mask_l维度是3维，添加通道维度等操作（根据实际网络输入要求调整）
#     #     mask_l = mask_l.unsqueeze(1)  # 在维度1的位置添加通道维度，假设符合网络对mask输入维度要求
#     # print('prep_input-mask_l-shape-2:', mask_l.shape)
#     # print('prep_input-mask_l-dtype:-2', mask_l.dtype)

#     return im_und_l, k_und_l, mask_l, im_gnd_l


def get_cine_mask(acc, acs_lines=4, x=18, y=192):
    """
    Generate a specific mask for CINE data.

    Parameters:
    acc: float - undersampling rate.
    acs_lines: int - number of autocalibration signal lines.
    x: int - width of the mask.
    y: int - height of the mask.
    """
    rows = y - acs_lines

    matrix = np.zeros((rows, x))

    ones_per_column = rows // acc  # y//acc-acs_lines

    first_column = np.zeros(rows)
    indices = np.linspace(0, rows - 1, ones_per_column, dtype=int)
    first_column[indices] = 1

    for j in range(x):
        matrix[:, j] = np.roll(first_column, j)

    insert_rows = np.ones((acs_lines, x))
    new_matrix = np.insert(matrix, rows // 2, insert_rows, axis=0)
    # print(new_matrix)

    # 这里根据实际需求决定是否保存mask为.mat文件，如果不需要可注释掉这行
    # mask_datadict = {'mask': np.squeeze(new_matrix)}
    # scio.savemat('/data0/huayu/Aluochen/Mypaper5/e_192x18_acs4_R4.mat', mask_datadict)

    # return new_matrix
    return new_matrix.astype(np.float64)  # 数据类型设为 float64 以匹配后续处理

# def prep_input(im, acc=4.0):
#     """Undersample the batch, then reformat them into what the network accepts.

#     Parameters
#     ----------
#     gauss_ivar: float - controls the undersampling rate.
#                         higher the value, more undersampling
#     """
#     mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
#     print('prep_input-mask-shape:',mask.shape)
#     print('prep_input-mask-dtype:',mask.dtype)
#     im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
#     im_gnd_l = torch.from_numpy(to_tensor_format(im))
#     im_und_l = torch.from_numpy(to_tensor_format(im_und))
#     k_und_l = torch.from_numpy(to_tensor_format(k_und))
#     mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))
#     print('prep_input-mask_l-shape:',mask_l.shape)
#     print('prep_input-mask_l-dtype:',mask_l.dtype)
#     return im_und_l, k_und_l, mask_l, im_gnd_l

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
