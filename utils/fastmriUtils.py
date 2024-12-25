import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
# from datasets.celeba import CelebA
# from datasets.ffhq import FFHQ
# from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np
# import sigpy.mri as mr
# import sigpy as sp
import random
# import mat73
import math
import re

import h5py
from torch.utils.data import Dataset, DataLoader
from utils.fastmriBaseUtils import *
# scipy.io 模块。sio 通常是 scipy.io 的别名，用于加载 MATLAB 文件（.mat 文件）。
import scipy.io as sio


'''
导入了必要的库，包括操作系统操作库os，PyTorch张量操作库torch，数字操作库numbers，
图像变换库torchvision.transforms，HDF5文件操作库h5py，以及一些自定义的模块和函数（例如utils）。 
注释掉的几行表明代码原本可能还依赖于sigpy库，但目前并未用到。
'''

'''
这段代码实现了一个用于加载和预处理快速MRI膝盖数据集的数据加载器，它支持多种数据预处理方式，并提供了灵活的配置选项。 
BaseUtils 模块中包含了 get_all_files, crop, IFFT2c, FFT2c 等函数，这些函数负责文件路径获取、图像裁剪、以及傅里叶变换等操作。
'''

'''
这是一个自定义的图像裁剪类，继承自object。它接受四个参数 x1, x2, y1, y2，分别表示裁剪区域的左上角和右下角坐标。
__call__ 方法使用 torchvision.transforms.functional.crop 函数进行图像裁剪。__repr__ 方法返回类的可读表示。
'''
class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class FastMRIKneeDataSet(Dataset):
    def __init__(self, maskpath, mode):
        super(FastMRIKneeDataSet, self).__init__()
        # self.config = config  # 配置对象，当前未使用

        if mode == 'train':
            # k-space 数据的目录路径
            self.kspace_dir = '/data0/chentao/data/fastMRI/multicoil_train/kspace/'
            # 灵敏度图的目录路径
            self.maps_dir = '/data0/chentao/data/fastMRI/multicoil_train/maps/'
            # 文件列表，包含指定目录中的前200个文件
            # self.file_list = get_all_files(self.kspace_dir)[:200]
            # self.file_list = get_all_files(self.kspace_dir)[:1]
            self.file_list = get_all_files(self.kspace_dir)[:500]
            # self.file_list = get_all_files(self.kspace_dir)[:100]
   
        elif mode == 'val':
            # 验证集的k-space数据目录
            self.kspace_dir = '/data0/taofeng/kneetest/T1data/'
            # 验证集的灵敏度图目录
            self.maps_dir = '/data0/taofeng/kneetest/csm/'
        elif mode == 'test':
            # 测试集的k-space数据目录
            # self.kspace_dir = '/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/T1_data_3/'
            # self.kspace_dir = '/data0/chentao/data/fastMRI/multicoil_train/test/'
            self.kspace_dir = '/data0/chentao/data/fastMRI/multicoil_train/kspace/'
            # 测试集的灵敏度图目录
            # self.maps_dir = '/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/Output_maps_3/'
            self.maps_dir = '/data0/chentao/data/fastMRI/multicoil_train/maps/'
            
            # 测试集文件列表，仅包含第一个文件
            self.file_list = get_all_files(self.kspace_dir)[:1]
        elif mode == 'datashift':
            # 数据迁移场景的k-space数据目录
            self.kspace_dir = '/data0/chentao/data/fastMRI_brain/brain_T2/'
            # 数据迁移场景的灵敏度图目录
            self.maps_dir = '/data0/chentao/data/fastMRI_brain/output_maps/'
        else:
            raise NotImplementedError

        # 数据集模式（训练、验证、测试等）
        self.mode = mode
        
        # 初始化每个文件的切片数量数组
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(
                self.kspace_dir, os.path.basename(file)))
            with h5py.File(os.path.join(self.kspace_dir, os.path.basename(file)), 'r') as data:
                if self.mode != 'sample':
                    # 非sample模式下，每个文件的切片数量减去5
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 5)
                else:
                    # sample模式下，使用全部切片
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # 创建累积索引用于映射
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # 从'0'开始计数

        # 加载掩码数据
        C = sio.loadmat(maskpath)    
        self.mask = C['mask'][:]
#       mask = np.transpose(mask, [1, 0])

    def __getitem__(self, idx):
        # 将索引转换为数值类型
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 确定当前索引对应的扫描编号
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # 计算在当前扫描中的切片索引
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] +
                self.num_slices[scan_idx] - 1)

        # sum(x0*conj(csm),1)
        #     csm_espirit = mr.app.EspiritCalib(ksp.squeeze().cpu().numpy(), calib_width=18, thresh=0.02, kernel_width=6, crop=0.95).run()
    
        # 注释掉的代码块：加载特定扫描和切片的灵敏度图 maps_idx就是图像域上的，不用IFFT2c
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            if self.mode != 'sample':
                slice_idx = slice_idx + 5
            maps_idx = data['s_maps'][slice_idx]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = crop(maps_idx, cropx=256, cropy=256)
            maps_idx = np.squeeze(maps_idx, 0)
            maps = np.asarray(maps_idx)

        # 加载特定扫描和切片的原始k空间数据
        raw_file = os.path.join(self.kspace_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            # if self.mode != 'sample':
            #     slice_idx = slice_idx + 5  # 非sample模式时，跳过前5帧
        
            # 获取k空间数据并进行预处理
            '''
            原始 k-space 数据：(15, 640, 368)
            增加一个维度：(1, 15, 640, 368)
            进行 IFFT2c 变换并裁剪：(1, 15, 320, 320)
            再次进行 FFT2c 变换：(1, 15, 320, 320)
            每一步变换后的 ksp_idx 维度如上所述
            '''
            ksp_idx = data['kspace'][slice_idx]  # 原始k空间数据，形状为15x640x368
            ksp_idx = np.expand_dims(ksp_idx, 0)  # 增加一个维度
            # ksp_idx = crop(IFFT2c(ksp_idx), cropx=320, cropy=320)  # 进行IFFT2c变换并裁剪
            # print(f"ksp_idx shape before IFFT2c: {ksp_idx.shape}")
            ksp_idx = crop(IFFT2c(ksp_idx), cropx=256, cropy=256)  # 进行IFFT2c变换并裁剪
            # print(f"ksp_idx shape after crop: {ksp_idx.shape}")
            # ksp_idx = t_crop(IFFT2c(ksp_idx), cropx=320, cropy=320)  # 进行IFFT2c变换并裁剪
            ksp_idx = FFT2c(ksp_idx)  # 再次进行FFT2c变换
            # print(f"ksp_idx shape after FFT2c: {ksp_idx.shape}")
            
            # 注释掉的归一化代码
            # if self.config.data.normalize_type == 'minmax':
            #     img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
            #     img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
            #     ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
            # elif self.config.data.normalize_type == 'std':
            #     minv = np.std(ksp_idx)*2
            #     ksp_idx = ksp_idx /  minv
            #     ksp_idx = np.squeeze(ksp_idx, 0)
            #     kspace = np.asarray(ksp_idx)

        # 获取图像数量
        nImg, *_ = ksp_idx.shape
    
        # 创建采样掩码
        # M = np.tile(self.mask,[nImg,30,1,1])  # 复制掩码以匹配数据维度
        M = np.tile(self.mask,[nImg,15,1,1])  # 复制掩码以匹配数据维度
        mask = self.mask.astype(np.complex64)  # 将掩码转换为复数类型
        mask = np.tile(mask, [nImg, 15, 1, 1])   
        # mask = np.tile(mask, [nImg, 30, 1, 1])   

        # 生成欠采样数据  
        # 返回原始的k空间数据、欠采样的k空间数据、归一化因子 return orgk(欠采样k空间),atb（欠采样K空间图像域图像）,minv 
        # ksp, orgk, minv = generateUndersampled(ksp_idx, mask)
        orgk, atb, minv = generateUndersampled(ksp_idx, mask)
        # 将k空间数据转换为实数表示
        # ksp = c2r(ksp)
        # orgk = c2r(orgk)

        # 去除多余的维度并转换为numpy数组
        ksp = np.asarray(np.squeeze(ksp_idx, 0))
        # orgk 存储了欠采样的 k-space 数据。k-space 是 MRI 数据的频域表示，包含了图像的频率信息。
        orgk = np.asarray(np.squeeze(orgk, 0))
        M = np.asarray(np.squeeze(M, 0))
        mask = np.asarray(np.squeeze(mask, 0))
        # return ksp, orgk, mask
        return ksp, orgk,maps

    def __len__(self):
        # 返回所有扫描的总切片数
        return int(np.sum(self.num_slices))
    
    
    ''''
    # mask = np.tile(mask, [nImg, 30, 1, 1])   会报下面的错：
    ---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[2], line 126
     54 # # from utils.utils_spirit import *
     55 # # 使用 csm_espirit 进行图像重建
     56 # # 假设你已经得到了 csm_espirit 和采样的 k-space 数据 kspace，可以使用以下代码进行图像重建：
   (...)
    114 # reconstructed_image_sense_np = reconstructed_image_sense.cpu().numpy()
    115 # reconstructed_image_cgSENSE_np = reconstructed_image_cgSENSE.cpu().numpy()
    116 
    117 # 定义r2c函数，用于将实数张量转换为复数张量
    118 def r2c(x):
   (...)
    123     return x
    124 
--> 126 for i,(ksp, orgk, mask) in enumerate(data_loader):
    127     # torch.Size([4, 30, 320, 320]) torch.Size([4, 30, 320, 320]) torch.Size([4, 15, 320, 320])
    128     # ksp-tpye-1: <class 'torch.Tensor'>
    129     # mask-tpye: <class 'torch.Tensor'>
    130     # ksp-tpye-2: <class 'torch.Tensor'>
    131     print(ksp.shape,orgk.shape,mask.shape)
    132     print('ksp-tpye-1:',type(ksp))

File ~/anaconda3/envs/k_gin/lib/python3.8/site-packages/torch/utils/data/dataloader.py:628, in _BaseDataLoaderIter.__next__(self)
    625 if self._sampler_iter is None:
...
    A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
  File "/data0/zhiyong/code/github/itzzy_git/k-gin-git/utils/fastmriBaseUtils.py", line 1776, in usp
    res=kspace[mask!=0]
IndexError: boolean index did not match indexed array along dimension 0; dimension is 15 but corresponding boolean dimension is 30
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
'''