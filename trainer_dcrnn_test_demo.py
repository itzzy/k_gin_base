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
# from scipy.io import loadmat
from scipy.io import savemat

from losses import CriterionKGIN
from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, NativeScalerWithGradNormCount as NativeScaler, add_weight_decay
from utils import multicoil2single
from utils import compressed_sensing as cs
from utils.dnn_io import to_tensor_format
from utils.dnn_io import from_tensor_format
from torch.autograd import Variable
from utils.metric import complex_psnr
from torch.cuda.amp import autocast, GradScaler


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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用 GPU 1 和 GPU 4

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
        
        self.model_name_run = config.general.model_name_run
        self.model_name_run_test =config.general.model_name_run_test

        self.start_epoch = 0
        self.only_infer = config.general.only_infer
        self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1
        # acc 加速倍速
        self.acc_rate = config.general.acc_rate
        # 通过索引取出列表中的元素（因为这里只有一个元素）并转换为整数类型
        self.acc_rate_value = int(self.acc_rate[0])
        print('acc_rate_value-type:',type(self.acc_rate_value))
        # print('self.acc_rate-dtype:',self.acc_rate.dtype)
        # 初始化 GradScaler
        # self.scaler = GradScaler()  # 用于混合精度训练
        # 替换弃用的 torch.cuda.amp 函数
        self.scaler = torch.amp.GradScaler('cuda')
        

        # data
        # train_ds = CINE2DT(config=config.data, mode='train')
        train_ds = CINE2DT(config=config.data, mode='val')
        test_ds = CINE2DT(config=config.data, mode='val')
        self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
                                       pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_ds, num_workers=2, drop_last=False, batch_size=1, shuffle=False)

        # network
        self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
        self.network.cuda()
        print("Parameter Count: %d" % count_parameters(self.network))

        # optimizer
        param_groups = add_weight_decay(self.network, config.training.optim_weight_decay)
        self.optimizer = eval(f'torch.optim.{config.optimizer.which}')(param_groups, **eval(f'config.optimizer.{config.optimizer.which}').__dict__)

        # 判断配置（config）中的 training.restore_ckpt 属性是否为 True。
        # 如果是 True，表示希望从之前保存的检查点恢复模型，那么就会调用 self.load_model 方法，
        # 并传入 config.training 作为参数，启动恢复模型的相关操作。
        if config.training.restore_training: self.load_model(config.training)
        self.loss_scaler = NativeScaler()

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
        start_time = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        self.network.train()
        train_err = 0
        train_batches = 0
        project_root = '.'
        save_dir_run = join(project_root, 'models/%s' % self.model_name_run)
        if not os.path.isdir(save_dir_run):
            os.makedirs(save_dir_run)
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
            # 检查是否是最后一个 epoch 的最后一个 batch
            is_last_epoch = (epoch == self.num_epochs - 1)
            is_last_batch = (i == len(self.train_loader) - 1)
            save_last = is_last_epoch and is_last_batch
            
            im_undersample, k_undersample, mask, im_groudtruth = prep_input(ref_img, self.acc_rate_value)
            im_undersample = Variable(im_undersample.type(Tensor))
            k_undersample = Variable(k_undersample.type(Tensor))
            mask = Variable(mask.type(Tensor))
            im_groudtruth = Variable(im_groudtruth.type(Tensor))
            # train_one_epoch-im_undersample torch.Size([2, 2, 192, 192, 18])
            # train_one_epoch-k_undersample torch.Size([2, 2, 192, 192, 18])
            # train_one_epoch-mask torch.Size([2, 2, 192, 192, 18])
            # train_one_epoch-im_groudtruth torch.Size([2, 2, 192, 192, 18])
            # print('train_one_epoch-im_undersample', im_undersample.shape)
            # print('train_one_epoch-k_undersample', k_undersample.shape)
            # print('train_one_epoch-mask', mask.shape)
            # print('train_one_epoch-im_groudtruth', im_groudtruth.shape)
            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            # with torch.cuda.amp.autocast(enabled=False):
            # with torch.no_grad():
            with autocast():
                im_recon = self.network(im_undersample, k_undersample,mask,test=False)  # size of kspace and mask: [B, T, H, W]
                # print('train_one_epoch-im_recon-loss',im_recon.requires_grad) 
                # train_one_epoch-im_recon torch.Size([2, 2, 192, 192, 18])
                # print('train_one_epoch-im_recon', im_recon.shape)
                # train_one_epoch-im_recon torch.Size([4, 18, 192, 192])
                # train_one_epoch-im_recon torch.Size([2, 2, 192, 192, 18])
                # train_one_epoch-im_recon-dtype: torch.float32
                # print('train_one_epoch-im_recon-dtype:', im_recon.dtype)
                loss = criterion(im_recon, im_groudtruth)
                # print('train_one_epoch-loss',loss.requires_grad)
            torch.cuda.empty_cache()
                # loss.backward()
                # self.optimizer.step()
            # 使用 GradScaler 进行梯度缩放
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            torch.cuda.empty_cache()

            train_err += loss.item()
            train_batches += 1
            running_loss += loss.item()
            epoch_loss = running_loss / train_batches if train_batches > 0 else 0
            # 判断当前 epoch 是否是 50 的倍数，如果是则打印平均损失
            if i % 20 == 0:
                print(f'Epoch {i} - Average Training Loss: {epoch_loss}')
            
            # 将损失写入TensorBoard
            # 注意：通常我们会在每个epoch结束时记录损失，但这里您可以根据需要调整
            self.writer.add_scalar('Training/Loss', loss.item(), epoch * len(self.train_loader) + i)
            self.logger.update_metric_item('train/recon_loss', loss.item())
            self.logger.update_metric_item('train/recon_loss_avg', loss.item()/len(self.train_loader))
            # 保存最后一个 epoch 和最后一个 batch 的数据
            if save_last:
                save_last_batch_data(im_undersample,k_undersample,mask,im_groudtruth,save_dir_run)
        # 在每个epoch结束时记录平均损失
        self.writer.add_scalar('Training/Average_Loss', epoch_loss, epoch)
    def run_test(self):
        # model_name = 'dc_rnn_test_demo'
        # # Configure directory info
        # project_root = '.'
        # self.save_dir = join(project_root, 'models/%s' % model_name)
        # if not os.path.isdir(self.save_dir):
        #     os.makedirs(self.save_dir)
        project_root = '.'
        save_dir_run_test = join(project_root, 'models/%s' % self.model_name_run_test)
        if not os.path.isdir(save_dir_run_test):
            os.makedirs(save_dir_run_test)
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

                # 准备输入数据
                im_und, k_und, mask, im_gnd = prep_input(ref_img, self.acc_rate_value)
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

                # 网络预测
                # im_recon = self.network(im_u, k_u, mask, test=False)
                # 使用 autocast 进行混合精度推理
                with autocast():
                    im_recon = self.network(im_u, k_u, mask, test=False)
                    # print('train_one_epoch-im_recon-test-loss:',im_recon.requires_grad)
                '''
                run_test-im_recon-shape: torch.Size([1, 2, 192, 192, 18])
                run_test-im_recon-dtype: torch.float32
                '''
                torch.cuda.empty_cache()
                # print('run_test-im_recon-shape:',im_recon.shape)
                # print('run_test-im_recon-dtype:',im_recon.dtype)
                im_recon_list.append(im_recon.cpu().data.numpy())  # 将 im_recon 转换为 numpy 数组并添加到列表中

                # 计算损失
                loss = criterion(im_recon, gnd)
                # print('train_one_epoch-im_recon-test-loss:',im_recon.requires_grad)
                running_test_loss += loss.item()
                torch.cuda.empty_cache()
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
                if i % 10 == 0:
                    epoch_test_loss = running_test_loss / test_batches if test_batches > 0 else 0
                    print(f"Batch {i} - Average Test Loss: {epoch_test_loss:.6f}")

            # 计算最终平均损失和 PSNR
            epoch_test_loss = running_test_loss / test_batches if test_batches > 0 else 0
            base_psnr /= (test_batches * self.test_loader.batch_size)
            test_psnr /= (test_batches * self.test_loader.batch_size)

            print(f"Final Test Loss: {epoch_test_loss:.6f}")
            print(f"Base PSNR: {base_psnr:.6f}")
            print(f"Test PSNR: {test_psnr:.6f}")
            
            # 将 im_recon 保存为.npy 文件
            im_recon_array = np.concatenate(im_recon_list, axis=0)  # 拼接所有 im_recon 张量
            np.save(join(save_dir_run_test, 'im_recon.npy'), im_recon_array)  # 保存为.npy 文件


        # 保存图像和模型
        i = 0
        for im_i, pred_i, und_i, mask_i in vis:
            im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
            plt.imsave(join(save_dir_run_test, f'im_{i}_x.png'), im, cmap='gray')

            im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                                    im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
            plt.imsave(join(save_dir_run_test, f'im_{i}_t.png'), im, cmap='gray')
            plt.imsave(join(save_dir_run_test, f'mask_{i}.png'),
            np.fft.fftshift(mask_i[..., 0]), cmap='gray')
            i += 1

        # 保存网络权重
        model_path = join(save_dir_run_test, "final_model.pth")
        torch.save(self.network.state_dict(), model_path)
        print(f"Model parameters saved at {model_path}")
        # 关闭TensorBoard
        self.writer.close()

def save_last_batch_data(im_undersample,k_undersample,mask,im_groudtruth,save_dir):
     # 将 Tensor 转换为 numpy 数组
    im_undersample_np = im_undersample.cpu().numpy()
    k_undersample_np = k_undersample.cpu().numpy()
    mask_np = mask.cpu().numpy()
    groudtruth_np = im_groudtruth.cpu().numpy()

    # 创建 train_output 子目录
    train_output_dir = join(save_dir, 'train_output')
    os.makedirs(train_output_dir, exist_ok=True)  # 如果目录不存在则创建

    # 保存为 .mat 格式
    savemat(join(train_output_dir, 'im_undersample.mat'), {'im_undersample': im_undersample_np})
    savemat(join(train_output_dir, 'k_undersample.mat'), {'k_undersample': k_undersample_np})
    savemat(join(train_output_dir, 'mask.mat'), {'mask': mask_np})
    savemat(join(train_output_dir, 'groudtruth.mat'), {'groudtruth': groudtruth_np})

    # 保存为 .npy 格式
    np.save(join(train_output_dir, 'im_undersample.npy'), im_undersample_np)
    np.save(join(train_output_dir, 'k_undersample.npy'), k_undersample_np)
    np.save(join(train_output_dir, 'mask.npy'), mask_np)
    np.save(join(train_output_dir, 'groudtruth.npy'), groudtruth_np)

    # 保存为 .png 格式
    plt.imsave(join(train_output_dir, 'im_undersample.png'), np.abs(im_undersample_np[0, 0, :, :, 0]), cmap='gray')
    plt.imsave(join(train_output_dir, 'mask.png'), np.abs(mask_np[0, 0, :, :, 0]), cmap='gray')
    plt.imsave(join(train_output_dir, 'groudtruth.png'), np.abs(groudtruth_np[0, 0, :, :, 0]), cmap='gray')

    # 将 k-space 数据转换到图像域并保存
    k_undersample_complex = k_undersample_np[0, 0, :, :, 0] + 1j * k_undersample_np[0, 1, :, :, 0]
    image_from_k_space = np.fft.ifft2(k_undersample_complex)
    image_from_k_space = np.abs(image_from_k_space)
    plt.imsave(join(train_output_dir, 'image_from_k_space.png'), image_from_k_space, cmap='gray')
    print(f"Saved im_undersample, k_undersample, mask, and groudtruth to {train_output_dir}")
    
    
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
    #             # train_one_epoch-ref_img torch.Size([2, 18, 192, 192])
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
    
    mask = np.expand_dims(mask, axis=0)  # 添加 batch 维度
    mask = np.expand_dims(mask, axis=0)  # 添加 time 维度
    # mask = np.tile(mask, (batch_size, time, 1, 1))  # 广播到完整形状
    # 得到的mask: (2, 192, 192, 18)
    mask = np.tile(mask, (batch_size, width, 1, 1))  # 广播到完整形状
    # AttributeError: 'numpy.ndarray' object has no attribute 'permute'
    # mask = mask.permute(0,3,2,1)
    # 将 NumPy 数组转换为 PyTorch 张量
    mask_tensor = torch.from_numpy(mask)

    # 使用 permute 方法重新排列维度
    mask_permuted = mask_tensor.permute(0, 3, 2, 1)
    # prep_input-mask_permuted-shape: torch.Size([2, 18, 192, 192])
    # print('prep_input-mask_permuted-shape:', mask_permuted.shape)
    
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