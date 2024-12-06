import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
#import tensorflow as tf
import logging
sqrt = np.sqrt
import torch.nn.functional as F
import torchvision.transforms as T
#import sigpy as sp
#from icecream import ic
from tqdm import tqdm
from scipy.linalg import null_space, svd
#from optimal_thresh import optht
import sigpy as sp
import sigpy.mri.app as MR
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupy import from_dlpack as cu_from_dlpack
#import pytorch_wavelets as wavelets

# class Aclass_sense:
#     def __init__(self, csm, mask, lam):
#         self.s = csm
#         self.mask = 1 - mask
#         self.lam = lam

#     def ATA(self, ksp):
#         Ax = sense(self.s, ksp)
#         AHAx = adjsense(self.s, Ax)
#         return AHAx

#     def A(self, ksp):
#         res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
#         return res


# def sense(csm, ksp):
#     """
#     :param csm: nb, nc, nx, ny 
#     :param ksp: nb, nc, nt, nx, ny
#     :return: SENSE output: nb, nt, nx, ny
#     """
#     # m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm),1) 
#     m = Emat_xyt(c2r(ksp), True, c2r(csm), 1)
#     res = Emat_xyt(m, False, c2r(csm), 1)   
#     # res  = fft2c_2d(m.unsqueeze(1) * csm)
#     return r2c(res) - ksp

# def adjsense(csm, ksp):
#     """
#     :param csm: nb, nc, nx, ny 
#     :param ksp: nb, nc, nt, nx, ny
#     :return: SENSE output: nb, nt, nx, ny
#     """
#     # m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm),1)    
#     # res  = fft2c_2d(m.unsqueeze(1) * csm)
#     m = Emat_xyt(c2r(ksp), True, c2r(csm), 1)
#     res = Emat_xyt(m, False, c2r(csm), 1) 
#     return r2c(res) - ksp


# class ConjGrad:
#     def __init__(self, A, rhs, max_iter=5, eps=1e-10):
#         self.A = A
#         self.b = rhs
#         self.max_iter = max_iter
#         self.eps = eps

#     def forward(self, x):
#         x = CG(x, self.b, self.A, max_iter=self.max_iter, eps=self.eps)
#         return x
    

# def dot_batch(x1, x2):
#     batch = x1.shape[0]
#     res = torch.reshape(x1 * x2, (batch, -1))
#     # res = torch.reshape(x1 * x2, (-1, 1))
#     return torch.sum(res, 1)


# def CG(x, b, A, max_iter, eps):
#     r = b - A.A(x)
#     p = r
#     rTr = dot_batch(torch.conj(r), r)
#     reshape = (-1,) + (1,) * (len(x.shape) - 1)
#     num_iter = 0
#     for iter in range(max_iter):
#         if rTr.abs().max() < eps:
#             break
#         Ap = A.A(p)
#         alpha = rTr / dot_batch(torch.conj(p), Ap)
#         alpha = torch.reshape(alpha, reshape)
#         x = x + alpha * p
#         r = r - alpha * Ap
#         rTrNew = dot_batch(torch.conj(r), r)
#         beta = rTrNew / rTr
#         beta = torch.reshape(beta, reshape)
#         p = r + beta * p
#         rTr = rTrNew

#         num_iter += 1
#     return x


# def cgSENSE(ksp, csm, mask, x0, niter, lam):
#     Aobj = Aclass_sense(csm, mask, lam)
#     y = - (1 - mask) * Aobj.ATA(ksp)
#     cg_iter = ConjGrad(Aobj, y, max_iter=niter)
#     x0 = Emat_xyt(x0, False, c2r(csm), 1)
#     x = cg_iter.forward(x=r2c(x0))
#     x = x * (1 - mask) + ksp
#     res = Emat_xyt(c2r(x), True, c2r(csm), 1)

#     return res

'''
get_mask_basic函数用于生成不同类型的掩码。
参数包括图像img、大小size、批次大小batch_size、掩码类型type、加速因子acc_factor、
中心分数center_fraction、是否固定fix、最小加速min_acc、线性权重linear_w、
线性密度linear_density和部分填充pf。
'''
def get_mask_basic(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False, min_acc=2, linear_w=1, linear_density=1, pf=1):
  # 计算采样点数
  mux_in = size ** 2
  if type.endswith('2d'):
    Nsamp = mux_in // acc_factor  # 2D采样点数
  elif type.endswith('1d'):
    Nsamp = size // acc_factor  # 1D采样点数

  # 生成不同类型的mask
  if type == 'gaussian2d':
    mask = torch.zeros_like(img)  # 初始化mask
    cov_factor = size * (1.5 / 128)  # 协方差因子
    mean = [size // 2, size // 2]  # 均值
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]  # 协方差矩阵
    if fix:
      samples = np.random.multivariate_normal(mean, cov, int(Nsamp))  # 生成多元正态分布样本
      int_samples = samples.astype(int)  # 转换为整数
      int_samples = np.clip(int_samples, 0, size - 1)  # 限制样本范围
      mask[..., int_samples[:, 0], int_samples[:, 1]] = 1  # 更新mask
    else:
      for i in range(batch_size):
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))  # 生成多元正态分布样本
        int_samples = samples.astype(int)  # 转换为整数
        int_samples = np.clip(int_samples, 0, size - 1)  # 限制样本范围
        mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1  # 更新mask
  elif type == 'uniformrandom2d':
    mask = torch.zeros_like(img)  # 初始化mask
    if fix:
      mask_vec = torch.zeros([1, size * size])  # 初始化mask向量
      samples = np.random.choice(size * size, int(Nsamp))  # 随机选择样本
      mask_vec[:, samples] = 1  # 更新mask向量
      mask_b = mask_vec.view(size, size)  # 重塑mask向量
      mask[:, ...] = mask_b  # 更新mask
    else:
      for i in range(batch_size):
        mask_vec = torch.zeros([1, size * size])  # 初始化mask向量
        samples = np.random.choice(size * size, int(Nsamp))  # 随机选择样本
        mask_vec[:, samples] = 1  # 更新mask向量
        mask_b = mask_vec.view(size, size)  # 重塑mask向量
        mask[i, ...] = mask_b  # 更新mask
  elif type == 'gaussian1d':
    mask = torch.zeros_like(img)  # 初始化mask
    mean = size // 2  # 均值
    std = size * (15.0 / 96)  # 标准差
    Nsamp_center = int(size * center_fraction)  # 中心采样点数
    if fix:
      samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))  # 生成正态分布样本
      int_samples = samples.astype(int)  # 转换为整数
      int_samples = np.clip(int_samples, 0, size - 1)  # 限制样本范围
      mask[... , int_samples] = 1  # 更新mask
      c_from = size // 2 - Nsamp_center // 2  # 中心区域起始位置
      mask[... , c_from:c_from + Nsamp_center] = 1  # 更新中心区域mask
    else:
      for i in range(batch_size):
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))  # 生成正态分布样本
        int_samples = samples.astype(int)  # 转换为整数
        int_samples = np.clip(int_samples, 0, size - 1)  # 限制样本范围
        mask[i, :, :, int_samples] = 1  # 更新mask
        c_from = size // 2 - Nsamp_center // 2  # 中心区域起始位置
        mask[i, :, :, c_from:c_from + Nsamp_center] = 1  # 更新中心区域mask
  elif type == 'uniform1d':
    mask = torch.zeros_like(img)  # 初始化mask
    if fix:
      Nsamp_center = int(size * center_fraction)  # 中心采样点数
      samples = np.random.choice(size, int(Nsamp - Nsamp_center))  # 随机选择样本
      mask[..., samples] = 1  # 更新mask
      c_from = size // 2 - Nsamp_center // 2  # 中心区域起始位置
      mask[..., c_from:c_from + Nsamp_center] = 1  # 更新中心区域mask
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)  # 中心采样点数
        samples = np.random.choice(size, int(Nsamp - Nsamp_center))  # 随机选择样本
        mask[i, :, :, samples] = 1  # 更新mask
        c_from = size // 2 - Nsamp_center // 2  # 中心区域起始位置
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1  # 更新中心区域mask
  elif type == 'regular1d':
    mask = torch.zeros_like(img)  # 初始化mask
    if fix:
      Nsamp_center = int(size * center_fraction)  # 中心采样点数
      samples = int(Nsamp - Nsamp_center)  # 样本数
      mask[..., 4:-1:acc_factor] = 1  # 更新mask
      c_from = size // 2 - Nsamp_center // 2  # 中心区域起始位置
      mask[..., c_from:c_from + Nsamp_center] = 1  # 更新中心区域mask
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)  # 中心采样点数
        samples = int(Nsamp - Nsamp_center)  # 样本数
        mask[i, :, :, 4:-1:acc_factor] = 1  # 更新mask
        c_from = size // 2 - Nsamp_center // 2  # 中心区域起始位置
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1  # 更新中心区域mask
  elif type == 'poisson':
    mask = poisson((img.shape[-2], img.shape[-1]), accel=acc_factor)  # 生成泊松分布mask
    mask = torch.from_numpy(mask)  # 转换为torch张量
  elif type == 'poisson1d':
    mask_pattern = abs(poisson((size, 2), accel=acc_factor)[:,1])  # 生成泊松分布mask模式
    mask = torch.zeros_like(img)  # 初始化mask
    mask[..., :] = torch.from_numpy(mask_pattern)  # 更新mask
  elif type == 'regularlinear':
    mask = torch.zeros_like(img)  # 初始化mask
    for i in range(batch_size):
      Nsamp_center_half = int(size * center_fraction/2)  # 中心采样点数的一半
      n_half = int(size/2)  # 尺寸的一半
      n_half_regular = n_half-Nsamp_center_half  # 规则采样点数的一半
      Nsamp_half = int(n_half_regular/acc_factor)  # 采样点数的一半
      Nsample_linear = int(Nsamp_half*linear_w)  # 线性采样点数
      Nsample_const = Nsamp_half - Nsample_linear  # 常数采样点数
      const_acc = round((n_half_regular-0.5*min_acc*Nsample_linear)/(Nsample_const + 0.5*Nsample_linear/linear_density))  # 常数加速度
      max_acc  = round(const_acc/linear_density)  # 最大加速度
      seg = max_acc - min_acc + 1  # 段数
      Nsamp_seg = int(Nsample_linear/seg)  # 每段采样点数
      if seg>1:
        Nsamp_seg_last = Nsamp_half-Nsamp_seg*(seg-1)  # 最后一段采样点数
      arr1 = [1];arr2 = [2]  # 初始化数组
      for j in range(min_acc,max_acc):
        for k in range(Nsamp_seg):
          arr1.append(arr1[-1] + j)  # 更新数组1
          arr2.append(arr2[-1] + j)  # 更新数组2
      if seg>1:
        for x in range(Nsamp_seg_last):
          if arr1[-1] + max_acc < int(n_half_regular*linear_w):
            arr1.append(arr1[-1] + max_acc)  # 更新数组1
          if arr2[-1] + max_acc < int(n_half_regular*linear_w):
            arr2.append(arr2[-1] + max_acc)  # 更新数组2
      while arr1[-1] + const_acc<n_half_regular:
        arr1.append(arr1[-1] + const_acc)  # 更新数组1
      while arr2[-1] + const_acc<n_half_regular:
        arr2.append(arr2[-1] + const_acc)  # 更新数组2
      mask_p1 = np.ones(size -2*n_half_regular)  # 初始化mask部分1
      mask_p2 = np.zeros(n_half_regular);mask_p2[arr1]=1  # 初始化mask部分2
      mask_p3 = np.zeros(n_half_regular);mask_p3[arr2]=1;mask_p3 = np.flip(mask_p3)  # 初始化mask部分3
      mask1d = np.concatenate((mask_p3,mask_p1,mask_p2))  # 合并mask部分
      mask = torch.zeros_like(img)  # 初始化mask
      mask[i, :, :, :] = torch.from_numpy(mask1d.astype(complex)).unsqueeze(0).unsqueeze(0).repeat(mask.shape[1],mask.shape[2],1)  # 更新mask
  else:
    NotImplementedError(f'Mask type {type} is currently not supported.')  # 不支持的mask类型
  
  # 处理pf参数
  if pf<1:
      pf_line = round(mask.shape[3]*(1-pf))  # 计算pf线
      mask[:, :, :, :pf_line] = 0  # 更新mask

  # 计算Nacc
  if type == 'poisson':
      Nacc = float(mask.shape[0]*mask.shape[1] / np.sum(abs(mask.cpu().numpy())))  # 计算Nacc
      mask = mask[None,None,:,:]  # 更新mask
  else:
      Nacc = float(mask.shape[3] / np.sum(abs(mask[0, 0, 0, :].cpu().numpy())))  # 计算Nacc
  
  # 生成mask1d
  mask1d = abs(mask[0,0,0,:].cpu().numpy()).astype(np.int32)  # 生成mask1d
  return mask,Nacc,mask1d  # 返回mask, Nacc, mask1d


import sigpy.mri as mr
# scout.squeeze().cpu().numpy()   得到的csm_espirit是numpy格式
# 这里需要输入一个(nx,ny,nc)的复数numpy k空间数据
# # 使用sigpy库的EspiritCalib函数对scout数据进行ESPIRiT校准，生成线圈灵敏度图csm_espirit。
# scout数据先通过squeeze去除维度，再转换为CPU上的numpy数组。calib_width=18表示校准区域的宽度，thresh=0.02表示阈值，kernel_width=6表示卷积核的宽度，crop=0.95表示裁剪比例。最后调用run()方法执行校准。
# csm_espirit = mr.app.EspiritCalib(scout.squeeze().cpu().numpy(), calib_width=18, thresh=0.02, kernel_width=6, crop=0.95).run()
# map 和csm_espirit
'''
这个函数用于对k空间数据进行ESPIRiT校准，生成线圈灵敏度图。
它接受k空间数据ksp、索引i、GPU ID和一些参数作为输入，
使用sigpy库的EspiritCalib函数在GPU上进行计算，返回计算得到的线圈灵敏度图csm。
'''
'''
这个函数用于对k空间数据进行ESPIRiT校准，生成线圈灵敏度图。
它接受k空间数据ksp、索引i、GPU ID和一些参数作为输入，
使用sigpy库的EspiritCalib函数在GPU上进行计算，返回计算得到的线圈灵敏度图csm。
参数：
- ksp: k空间数据
- i: 需要处理的切片索引
- gpu_id: GPU的ID
- calib: 校准区域的宽度，默认为24
- crop: 裁剪比例，默认为0
'''
def ESPIRiT_calib(ksp, i, gpu_id, calib=24, crop=0):
    # 将k空间数据ksp的第i个切片去除多余的维度
    kdata = torch.squeeze(ksp[i]) 
    # 将kdata转换为GPU上的数据
    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    # 使用sigpy库的EspiritCalib函数进行ESPIRiT校准，生成线圈灵敏度图csm
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    # 将 csm 从 GPU 数据转换回 PyTorch 张量。这里使用了 toDlpack 函数将 csm 转换为 DLPack 格式，
    # 然后使用 from_dlpack 函数将 DLPack 数据转换为 PyTorch 张量。
    csm = from_dlpack(csm.toDlpack())
    # 返回计算得到的线圈灵敏度图csm
    return csm


'''
这是ESPIRiT_calib的并行版本，主要区别是它处理整个kspace数据，而不是单个切片。返回的csm多了一个维度。
参数：
- ksp: k空间数据
- gpu_id: GPU的ID
- calib: 校准区域的宽度，默认为24
- crop: 裁剪比例，默认为0
'''
def ESPIRiT_calib_parallel(ksp, gpu_id, calib=24, crop=0):
    # 将k空间数据ksp去除多余的维度
    kdata = torch.squeeze(ksp) 
    # 将kdata转换为GPU上的数据
    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    # 使用sigpy库的EspiritCalib函数进行ESPIRiT校准，生成线圈灵敏度图csm
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    # 将 csm 从 GPU 数据转换回 PyTorch 张量。这里使用了 toDlpack 函数将 csm 转换为 DLPack 格式，
    # 然后使用 from_dlpack 函数将 DLPack 数据转换为 PyTorch 张量。
    csm = from_dlpack(csm.toDlpack())
    # 返回计算得到的线圈灵敏度图csm，并增加一个维度
    return csm.unsqueeze(0)

'''
这个函数用于对预扫描的k空间数据进行ESPIRiT校准。
它接受预扫描的k空间数据ksp_prescan、实际k空间数据kspace、索引i、GPU ID和一些参数作为输入，
返回计算得到的线圈灵敏度图csm。
'''
def ESPIRiT_calib_prescan(ksp_prescan, ksp, i, gpu_id, calib=24, crop=0):
    kdata = torch.squeeze(ksp_prescan[i]) 
    calib = kdata.shape[-1]
    zpad = T.CenterCrop((int(ksp.shape[-2]),int(ksp.shape[-1])))
    kdata = zpad(kdata)

    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    csm = from_dlpack(csm.toDlpack())
    return csm

# 设置随机数种子，确保可重复性。
def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True  # 固定卷积算法, 设为True会导致卷积变慢
        torch.backends.cudnn.benchmark = False

# 将数据保存到MAT文件
def save_mat(save_dict, variable, file_name, index=0, Complex=True, normalize=True):
    # variable = variable.cpu().detach().numpy()
    if normalize:

        if Complex:
            variable = normalize_complex(variable)
        else:
            variable_abs = torch.abs(variable)
            coeff = torch.max(variable_abs)
            variable = variable / coeff
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) +
                        '_' + str(index + 1) + '.mat')
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)


def hfssde_save_mat(config, variable, variable_name='recon', normalize=True):
    if normalize:
        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    save_dict = config.sampling.folder
    file_name = config.training.sde + '_acc' + config.sampling.acc + '_acs' + config.sampling.acs \
                    + '_epoch' + str(config.sampling.ckpt)
    file = os.path.join(save_dict, str(file_name) + '.mat')
    datadict = {variable_name: np.squeeze(variable)}
    scio.savemat(file, datadict)


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

# 将字典转换为命名空间对象，方便命令行参数解析
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# 将复数数据转换为张量
def to_tensor(x):
    # 提取实部
    re = np.real(x)
    # 提取虚部
    im = np.imag(x)
    # 沿第二个维度（通道维度）拼接实部和虚部
    x = np.concatenate([re, im], 1)
    # 删除临时变量以释放内存
    del re, im
    # 将NumPy数组转换为PyTorch张量并返回
    return torch.from_numpy(x)


# 对图像进行中心裁剪，支持4D和3D图像
def spirit_crop(img, cropc, cropx, cropy):
    if img.ndim == 4:  # 如果是4D图像
        nb, c, x, y = img.shape
        # 计算裁剪的起始位置
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        # 执行裁剪
        cimg = img[:, startc:startc + cropc, startx:startx + cropx, starty: starty + cropy]
    elif img.ndim == 3:  # 如果是3D图像
        c, x, y = img.shape
        # 计算裁剪的起始位置
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        # 执行裁剪
        cimg = img[startc:startc + cropc, startx:startx + cropx, starty: starty + cropy]
    
    return cimg

# 对图像进行中心裁剪，主要用于4D图像
def crop(img, cropx, cropy=None):
    nb, c, y, x = img.shape
    # 如果未指定cropy，则使用y的值
    if cropy==None:
        cropy=y    
    # 计算裁剪的起始位置
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    # 执行裁剪并返回结果
    return img[:, :, starty:starty + cropy, startx:startx + cropx]

# def t_crop(img, cropx, cropy):
#     """
#     对图像进行中心裁剪
#     :param img: 输入图像，形状为 (N, C, H, W)
#     :param cropx: 裁剪后的宽度
#     :param cropy: 裁剪后的高度
#     :return: 裁剪后的图像
#     """
#     nb, c, x, y = img.size()
#     startx = x // 2 - cropx // 2
#     starty = y // 2 - cropy // 2
#     cimg = img[:, :, startx:startx + cropx, starty: starty + cropy]
    
#     return cimg

'''
如果 img 是一个 numpy 数组，那么调用 img.size() 会报错。
numpy 数组没有 size() 方法，而是有一个 size 属性，它返回数组的总元素个数。
# 为了获取 numpy 数组的形状，你应该使用 img.shape 属性。
'''
def t_crop(img, cropx, cropy):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected img to be a numpy array, but got {type(img)}")

    nb, c, x, y = img.size()
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    cimg = img[:, :, startx:startx + cropx, starty: starty + cropy]
    

# 定义acs_crop函数，用于裁剪图像的中心区域
def acs_crop(img, cropx, cropy):
    # 创建一个与输入图像相同大小的全零张量
    acs = torch.zeros_like(img)
    # 获取输入图像的尺寸
    nb, c, x, y = img.size()
    # 计算裁剪的起始x坐标
    startx = x // 2 - cropx // 2
    # 计算裁剪的起始y坐标
    starty = y // 2 - cropy // 2
    # 将输入图像的中心区域复制到acs张量中
    acs[:, :, startx:startx + cropx, starty: starty + cropy] = img[:, :, startx:startx + cropx, starty: starty + cropy]
    
    # 返回裁剪后的结果
    return acs

# 定义inv_crop函数，用于将小尺寸张量填充到大尺寸张量中
def inv_crop(target, center_tensor):
    # 创建一个与目标尺寸相同的全零张量
    padded_tensor = torch.zeros_like(target)
    # 计算各个维度的填充大小
    pad_top = (padded_tensor.shape[0] - center_tensor.shape[0]) // 2
    pad_bottom = padded_tensor.shape[0] - center_tensor.shape[0] - pad_top
    pad_left = (padded_tensor.shape[1] - center_tensor.shape[1]) // 2
    pad_right = padded_tensor.shape[1] - center_tensor.shape[1] - pad_left
    pad_front = (padded_tensor.shape[2] - center_tensor.shape[2]) // 2
    pad_back = padded_tensor.shape[2] - center_tensor.shape[2] - pad_front
    pad_leftmost = (padded_tensor.shape[3] - center_tensor.shape[3]) // 2
    pad_rightmost = padded_tensor.shape[3] - center_tensor.shape[3] - pad_leftmost

    # 使用 pad 函数进行填充
    padded_tensor = F.pad(center_tensor, (pad_leftmost, pad_rightmost, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom))
    return padded_tensor

# 定义inv_crop_numpy函数，用于NumPy数组的填充操作
def inv_crop_numpy(target, tensor):
    # 获取目标尺寸和输入张量的尺寸
    target_size = target.shape
    tensor_shape = np.array(tensor.shape)
    target_size = np.array(target_size)
    # 计算需要填充的大小
    pad_sizes = np.maximum(target_size - tensor_shape, 0)
    pad_left = pad_sizes // 2
    pad_right = pad_sizes - pad_left
    # 创建填充配置
    padding = [(pad_left[i], pad_right[i]) for i in range(len(tensor_shape))]
    # 使用numpy的pad函数进行填充
    padded_tensor = np.pad(tensor, padding, mode='constant')
    return padded_tensor

# 定义torch_crop函数，用于处理不同情况下的图像裁剪和填充
def torch_crop(img, cropx, cropy):
    # 获取输入图像的尺寸
    nb, c, x, y = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    # 根据不同情况进行裁剪或填充
    if y > cropy and x > cropx:
        img = crop(img, cropx, cropy)
    elif y > cropy and x < cropx:
        img = crop(img, x, cropy)
        target = torch.zeros(nb,c,cropx,cropy)
        img = inv_crop(target,img)
    elif y < cropy and x > cropx:
        img = crop(img, cropx, y)
        target = torch.zeros(nb,c,cropx,cropy)
        img = inv_crop(target,img)
    else:
        target = torch.zeros(nb,c,cropx,cropy)
        img = inv_crop(target,img)
    return img

# 定义pad_or_crop_tensor函数，用于将输入张量填充或裁剪到目标形状
def pad_or_crop_tensor(input_tensor, target_shape):
    input_shape = input_tensor.shape
    pad_width = []

    # 计算每个维度需要填充的宽度或裁剪的宽度
    for i in range(len(target_shape)):
        diff = target_shape[i] - input_shape[i]
        # 计算前后需要填充的宽度或裁剪的宽度
        pad_before = max(0, diff // 2)
        pad_after = max(0, diff - pad_before)
        pad_width.append((pad_before, pad_after))

    # 使用numpy的pad函数进行填充或裁剪
    padded_tensor = np.pad(input_tensor, pad_width, mode='constant')

    # 裁剪张量为目标形状
    cropped_tensor = padded_tensor[:target_shape[0], :target_shape[1], :target_shape[2], :target_shape[3]]

    return cropped_tensor

# 定义normalize函数，用于将图像归一化到[0, 1]范围
def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img

# 定义normalize_np函数，用于NumPy数组的归一化
def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

# 定义normalize_complex函数，用于复数图像的归一化
def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)

# 定义normalize_l2函数，用于L2归一化
def normalize_l2(img):
    minv = np.std(img)
    img = img / minv
    return img

# 定义get_data_scaler函数，用于获取数据缩放器
def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x

# 定义get_data_inverse_scaler函数，用于获取数据逆缩放器
def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x

# 定义get_mask函数，用于获取采样掩码
def get_mask(config, caller):
    # 根据不同的调用者和配置选择掩码文件
    if caller == 'sde':
        if config.training.mask_type == 'low_frequency':
            mask_file = 'mask/' +  config.training.mask_type + "_acs" + config.training.acs + '.mat'
        elif config.training.mask_type == 'center':
            mask_file = 'mask/' +  config.training.mask_type + "_length" + config.training.acs + '.mat'
        else:
            mask_file = 'mask/' +  config.training.mask_type + "_acc" + config.training.acc \
                                                + '_acs' + config.training.acs + '.mat'
    elif caller == 'sample':
        mask_file = 'mask/' +  config.sampling.mask_type + "_acc" + config.sampling.acc \
                                                + '_acs' + config.sampling.acs + '.mat'
    elif caller == 'acs':
        mask_file = 'mask/low_frequency_acs18.mat'
    # 加载掩码文件
    mask = scio.loadmat(mask_file)['mask']
    # 将掩码转换为复数类型
    mask = mask.astype(np.complex)
    # 扩展掩码的维度
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    # 将掩码转换为PyTorch张量并移动到指定设备
    mask = torch.from_numpy(mask).to(config.device)

    return mask
# def get_mask(config, caller):
#     if caller == 'sde':
#         if config.training.mask_type == 'low_frequency':
#             mask_file = 'mask/' +  config.training.mask_type + "_acs" + config.training.acc + '.mat'
#         else:
#             mask_file = 'mask_acs20/' +  config.training.mask_type + "_acc" + config.training.acc + '.mat'
#     elif caller == 'sample':
#         mask_file = 'mask_acs18/' +  config.sampling.mask_type + "_acc" + config.sampling.acc + '.mat'
#     mask = scio.loadmat(mask_file)['mask']
#     mask = mask.astype(np.complex128)
#     mask = np.expand_dims(mask, axis=0)
#     mask = np.expand_dims(mask, axis=0)
#     mask = torch.from_numpy(mask).to(config.device)

#     return mask


# 定义ifftshift函数，用于对张量进行逆傅里叶变换的频移操作
def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True  # 确保输入x是一个PyTorch张量
    if axes is None:
        axes = tuple(range(x.ndim))  # 如果未指定轴，则使用所有维度
        shift = [-(dim // 2) for dim in x.shape]  # 计算每个维度的移动量
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)  # 如果axes是整数，计算单个维度的移动量
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]  # 计算指定轴的移动量
    return torch.roll(x, shift, axes)  # 使用torch.roll进行张量移动


# 定义fftshift函数，用于对张量进行傅里叶变换的频移操作
def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True  # 确保输入x是一个PyTorch张量
    if axes is None:
        axes = tuple(range(x.ndim()))  # 如果未指定轴，则使用所有维度
        shift = [dim // 2 for dim in x.shape]  # 计算每个维度的移动量
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2  # 如果axes是整数，计算单个维度的移动量
    else:
        shift = [x.shape[axis] // 2 for axis in axes]  # 计算指定轴的移动量
    return torch.roll(x, shift, axes)  # 使用torch.roll进行张量移动


# 定义fft2c函数，用于执行2D傅里叶变换
def fft2c(x):
    device = x.device  # 获取输入张量的设备
    nb, nc, nt, nx, ny = x.size()  # 获取输入张量的维度
    ny = torch.Tensor([ny]).to(device)  # 将ny转换为张量并移动到相应设备
    nx = torch.Tensor([nx]).to(device)  # 将nx转换为张量并移动到相应设备
    x = ifftshift(x, axes=3)  # 对第4个维度进行ifftshift操作
    x = torch.transpose(x, 3, 4)  # 交换第4和第5个维度
    x = FFT.fft(x)  # 执行1D FFT
    x = torch.transpose(x, 3, 4)  # 交换回第4和第5个维度
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))  # 对结果进行fftshift并除以sqrt(nx)
    x = ifftshift(x, axes=4)  # 对第5个维度进行ifftshift操作
    x = FFT.fft(x)  # 执行1D FFT
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))  # 对结果进行fftshift并除以sqrt(ny)
    return x


# 定义fft2c_2d函数，用于执行2D傅里叶变换（针对4D输入）
def fft2c_2d(x):
    device = x.device  # 获取输入张量的设备
    nb, nc, nx, ny = x.size()  # 获取输入张量的维度
    ny = torch.Tensor([ny]).to(device)  # 将ny转换为张量并移动到相应设备
    nx = torch.Tensor([nx]).to(device)  # 将nx转换为张量并移动到相应设备
    x = ifftshift(x, axes=2)  # 对第3个维度进行ifftshift操作
    x = torch.transpose(x, 2, 3)  # 交换第3和第4个维度
    x = FFT.fft(x)  # 执行1D FFT
    x = torch.transpose(x, 2, 3)  # 交换回第3和第4个维度
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))  # 对结果进行fftshift并除以sqrt(nx)
    x = ifftshift(x, axes=3)  # 对第4个维度进行ifftshift操作
    x = FFT.fft(x)  # 执行1D FFT
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))  # 对结果进行fftshift并除以sqrt(ny)
    return x


# 定义FFT2c函数，用于执行2D傅里叶变换（使用NumPy）
def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)  # 获取输入数组的维度
    x = np.fft.ifftshift(x, axes=2)  # 对第3个维度进行ifftshift操作
    x = np.transpose(x, [0, 1, 3, 2])  # 交换第3和第4个维度
    x = np.fft.fft(x, axis=-1)  # 沿最后一个轴执行1D FFT
    x = np.transpose(x, [0, 1, 3, 2])  # 交换回第3和第4个维度
    x = np.fft.fftshift(x, axes=2)/np.math.sqrt(nx)  # 对结果进行fftshift并除以sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)  # 对第4个维度进行ifftshift操作
    x = np.fft.fft(x, axis=-1)  # 沿最后一个轴执行1D FFT
    x = np.fft.fftshift(x, axes=3)/np.math.sqrt(ny)  # 对结果进行fftshift并除以sqrt(ny)
    return x

'''
`ifft2c` 函数和 `IFFT2c` 函数的主要区别在于它们的实现方式和使用的库。以下是对这两个函数的详细比较：

### 1. 实现方式

- **`ifft2c` 函数**：
  - 使用 **PyTorch** 库实现，适用于 GPU 加速。
  - 处理的是一个五维张量，通常用于深度学习模型中的数据。
  - 通过 `FFT.ifft` 函数执行逆傅里叶变换，支持在 GPU 上进行计算。
  - 具有设备管理功能，能够自动处理输入张量所在的设备（CPU 或 GPU）。

```python
def ifft2c(x):
    device = x.device  # 获取输入张量的设备
    nb, nc, nt, nx, ny = x.size()  # 获取输入张量的维度
    # ... 进行一系列的逆傅里叶变换和维度调整
    return x
```

- **`IFFT2c` 函数**：
  - 使用 **NumPy** 库实现，适用于 CPU 计算。
  - 处理的是一个四维数组，通常用于数据处理和分析。
  - 通过 `np.fft.ifft` 函数执行逆傅里叶变换，适合于较小的数据集。
  - 不支持 GPU 加速，所有计算都在 CPU 上进行。

```python
def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)  # 获取输入数组的维度
    # ... 进行一系列的逆傅里叶变换和维度调整
    return x
```

### 2. 输入数据的维度

- **`ifft2c`**：
  - 处理的输入数据是一个五维张量，形状为 `(nb, nc, nt, nx, ny)`，其中 `nb` 是批次大小，`nc` 是通道数，`nt` 是时间维度，`nx` 和 `ny` 是空间维度。

- **`IFFT2c`**：
  - 处理的输入数据是一个四维数组，形状为 `(nb, nc, nx, ny)`，没有时间维度。

### 3. 计算性能

- **`ifft2c`**：
  - 由于使用了 PyTorch，能够利用 GPU 加速，适合处理大规模数据和深度学习任务。

- **`IFFT2c`**：
  - 由于使用了 NumPy，计算性能较低，适合处理较小的数据集。

### 4. 设备管理

- **`ifft2c`**：
  - 自动获取输入张量的设备（CPU 或 GPU），并在相应设备上执行计算。

- **`IFFT2c`**：
  - 仅在 CPU 上执行计算，不支持 GPU。

### 总结

- **`ifft2c`** 是一个基于 PyTorch 的函数，适合于深度学习和大规模数据处理，支持 GPU 加速，处理五维张量。
- **`IFFT2c`** 是一个基于 NumPy 的函数，适合于数据分析和处理，主要在 CPU 上执行，处理四维数组。

如果你有更多具体的问题或需要进一步的解释，请告诉我。
'''


'''
在处理 MRI 数据时，`nt` 通常指的是时间维度（time dimension），它表示在时间序列中采集的不同时间点的图像或数据。以下是对时间维度的详细解释：
### 时间维度（nt）
1. **定义**：
   - 在 MRI 中，时间维度 `nt` 表示在不同时间点上采集的 k-space 数据或图像数据。它可以用于动态成像（dynamic imaging）或时间序列分析。

2. **应用场景**：
   - **动态 MRI**：在动态 MRI 中，患者在扫描过程中可能会进行某种活动（如心跳、呼吸等），因此在不同的时间点上采集图像数据。时间维度 `nt` 允许我们捕捉这些变化。
   - **功能性 MRI（fMRI）**：在功能性 MRI 中，时间维度用于捕捉大脑活动的变化，通常在多个时间点上进行扫描，以观察大脑在不同刺激下的反应。

3. **数据结构**：
   - 在一个五维张量中，形状为 `(nb, nc, nt, nx, ny)`：
     - `nb`：批次大小（number of batches），表示一次处理的样本数量。
     - `nc`：通道数（number of channels），通常与接收线圈的数量相关。
     - `nt`：时间维度，表示在不同时间点上采集的数据。
     - `nx` 和 `ny`：空间维度，表示图像的宽度和高度。

### 示例

假设你有一个 MRI 数据集，其中包含在不同时间点上采集的图像数据。数据的形状可能是 `(10, 8, 30, 256, 256)`，其中：
- `10` 是批次大小（nb），表示一次处理 10 个样本。
- `8` 是通道数（nc），表示使用 8 个接收线圈。
- `30` 是时间维度（nt），表示在 30 个不同时间点上采集的数据。
- `256` 和 `256` 是空间维度（nx 和 ny），表示图像的宽度和高度。

### 总结

时间维度 `nt` 在 MRI 数据中用于表示在不同时间点上采集的图像或数据，适用于动态成像和功能性 MRI 等应用场景。它允许研究人员和医生观察和分析随时间变化的生理过程。

如果你有更多具体的问题或需要进一步的解释，请告诉我。
'''

# 定义ifft2c函数，用于执行2D逆傅里叶变换
def ifft2c(x):
    device = x.device  # 获取输入张量的设备
    nb, nc, nt, nx, ny = x.size()  # 获取输入张量的维度
    ny = torch.Tensor([ny])
    ny = ny.to(device)  # 将ny转换为张量并移动到相应设备
    nx = torch.Tensor([nx])
    nx = nx.to(device)  # 将nx转换为张量并移动到相应设备
    x = ifftshift(x, axes=3)  # 对第4个维度进行ifftshift操作
    x = torch.transpose(x, 3, 4)  # 交换第4和第5个维度
    x = FFT.ifft(x)  # 执行1D IFFT
    x = torch.transpose(x, 3, 4)  # 交换回第4和第5个维度
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(nx))  # 对结果进行fftshift并乘以sqrt(nx)
    x = ifftshift(x, axes=4)  # 对第5个维度进行ifftshift操作
    x = FFT.ifft(x)  # 执行1D IFFT
    x = torch.mul(fftshift(x, axes=4), torch.sqrt(ny))  # 对结果进行fftshift并乘以sqrt(ny)
    return x


# 定义ifft2c_2d函数，用于执行2D逆傅里叶变换（针对4D输入）
def ifft2c_2d(x):
    device = x.device  # 获取输入张量的设备
    nb, nc, nx, ny = x.size()  # 获取输入张量的维度
    ny = torch.Tensor([ny])
    ny = ny.to(device)  # 将ny转换为张量并移动到相应设备
    nx = torch.Tensor([nx])
    nx = nx.to(device)  # 将nx转换为张量并移动到相应设备
    x = ifftshift(x, axes=2)  # 对第3个维度进行ifftshift操作
    x = torch.transpose(x, 2, 3)  # 交换第3和第4个维度
    x = FFT.ifft(x)  # 执行1D IFFT
    x = torch.transpose(x, 2, 3)  # 交换回第3和第4个维度
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))  # 对结果进行fftshift并乘以sqrt(nx)
    x = ifftshift(x, axes=3)  # 对第4个维度进行ifftshift操作
    x = FFT.ifft(x)  # 执行1D IFFT
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))  # 对结果进行fftshift并乘以sqrt(ny)
    return x
'''
使用 `ifft2c_2d` 函数或 `IFFT2c` 函数的选择主要取决于输入数据的维度和你希望使用的计算库。
以下是对这两个函数的详细比较，以帮助你理解为什么在这种情况下更适合使用 `ifft2c_2d`。
### 1. 输入数据的维度
- **`ifft2c_2d`**：
  - 该函数设计用于处理四维输入数据，形状为 `(N, C, H, W)`，其中：
    - `N` 是批次大小（number of batches）。
    - `C` 是通道数（number of channels）。
    - `H` 是图像的高度（ny）。
    - `W` 是图像的宽度（nx）。
  - 适合处理你提供的 k-space 数据形状 `(36, 15, 640, 356)`。

- **`IFFT2c`**：
  - 该函数设计用于处理四维 NumPy 数组，形状为 `(nb, nc, nx, ny)`，其中：
    - `nb` 是批次大小。
    - `nc` 是通道数。
    - `nx` 和 `ny` 是空间维度。
  - 适合处理 NumPy 数组，而不是 PyTorch 张量。

### 2. 计算库

- **`ifft2c_2d`**：
  - 使用 **PyTorch** 库，能够利用 GPU 加速，适合深度学习和大规模数据处理。
  - 适合在 PyTorch 环境中进行计算，特别是当你已经在使用 PyTorch 处理数据时。

- **`IFFT2c`**：
  - 使用 **NumPy** 库，适合在 CPU 上进行计算。
  - 如果你的数据是 NumPy 数组，且不需要 GPU 加速，可以使用这个函数。

### 3. 选择的理由

- **在你的情况下**，由于 k-space 数据的形状为 `(36, 15, 640, 356)`，并且假设你在使用 PyTorch 进行处理，使用 `ifft2c_2d` 函数是更合适的选择，因为它能够直接处理 PyTorch 张量，并且支持 GPU 加速。

- 如果你使用的是 NumPy 数组，并且不需要 GPU 加速，那么可以选择 `IFFT2c` 函数。

### 总结

- 使用 `ifft2c_2d` 函数是因为它适合处理形状为 `(36, 15, 640, 356)` 的四维 PyTorch 张量。
- 使用 `IFFT2c` 函数适合处理 NumPy 数组，且不需要 GPU 加速。

如果你有更多具体的问题或需要进一步的解释，请告诉我。
'''

# 定义IFFT2c函数，用于执行2D逆傅里叶变换（使用NumPy）
def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)  # 获取输入数组的维度
    x = np.fft.ifftshift(x, axes=2)  # 对第3个维度进行ifftshift操作
    x = np.transpose(x, [0, 1, 3, 2])  # 交换第3和第4个维度
    x = np.fft.ifft(x, axis=-1)  # 沿最后一个轴执行1D IFFT
    x = np.transpose(x, [0, 1, 3, 2])  # 交换回第3和第4个维度
    x = np.fft.fftshift(x, axes=2)*np.math.sqrt(nx)  # 对结果进行fftshift并乘以sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)  # 对第4个维度进行ifftshift操作
    x = np.fft.ifft(x, axis=-1)  # 沿最后一个轴执行1D IFFT
    x = np.fft.fftshift(x, axes=3)*np.math.sqrt(ny)  # 对结果进行fftshift并乘以sqrt(ny)
    return x


'''
根据你提供的 k-space 数据的形状 `(36, 15, 640, 356)`，我们可以分析这个形状的含义：

- `36`：表示批次大小（number of batches），即一次处理的样本数量。
- `15`：表示通道数（number of channels），通常与接收线圈的数量相关。
- `640`：表示图像的高度（ny）。
- `356`：表示图像的宽度（nx）。

### 选择合适的函数

根据这个形状，你应该使用 `ifft2c_2d` 函数，因为它是专门为处理四维输入（形状为 `(N, C, H, W)`）设计的。`ifft2c_2d` 函数的定义如下：

```python
def ifft2c_2d(x):
    device = x.device  # 获取输入张量的设备
    nb, nc, nx, ny = x.size()  # 获取输入张量的维度
    # ... 进行一系列的逆傅里叶变换和维度调整
    return x
```

### 使用示例

以下是如何使用 `ifft2c_2d` 函数处理你的 k-space 数据的示例代码：

```python
import torch

# 假设 kspace 是你的 k-space 数据，形状为 (36, 15, 640, 356)
kspace = torch.randn(36, 15, 640, 356) + 1j * torch.randn(36, 15, 640, 356)  # 示例数据

# 调用 ifft2c_2d 函数进行逆傅里叶变换
image_space = ifft2c_2d(kspace)

# 打印结果
print("逆傅里叶变换后的图像形状:", image_space.shape)
```

### 总结

- 使用 `ifft2c_2d` 函数来处理形状为 `(36, 15, 640, 356)` 的 k-space 数据。
- 该函数将执行 2D 逆傅里叶变换，并返回图像域的数据。

如果你有更多具体的问题或需要进一步的解释，请告诉我。
'''

# 定义Emat_xyt函数，用于执行正向和反向的傅里叶变换操作
def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask  # 将实数转换为复数并应用掩码
            if b.ndim == 4:
                b = ifft2c_2d(b)  # 执行2D IFFT（4D输入）
            else:
                b = ifft2c(b)  # 执行2D IFFT（5D输入）
            x = c2r(b)  # 将复数转换回实数
        else:
            b = r2c(b)  # 将实数转换为复数
            if b.ndim == 4:
                b = fft2c_2d(b) * mask  # 执行2D FFT并应用掩码（4D输入）
            else:
                b = fft2c(b) * mask  # 执行2D FFT并应用掩码（5D输入）
            x = c2r(b)  # 将复数转换回实数
    else:
        if inv:
            csm = r2c(csm)  # 将灵敏度图转换为复数
            x = r2c(b) * mask  # 将输入转换为复数并应用掩码
            if b.ndim == 4:
                x = ifft2c_2d(x)  # 执行2D IFFT（4D输入）
            else:
                x = ifft2c(x)  # 执行2D IFFT（5D输入）
            x = x*torch.conj(csm)  # 应用灵敏度图的共轭
            x = torch.sum(x, 1)  # 沿通道维度求和
            x = torch.unsqueeze(x, 1)  # 添加通道维度
            x = c2r(x)  # 将复数转换回实数
        else:
            csm = r2c(csm)  # 将灵敏度图转换为复数
            b = r2c(b)  # 将输入转换为复数
            b = b*csm  # 应用灵敏度图
            if b.ndim == 4:
                b = fft2c_2d(b)  # 执行2D FFT（4D输入）
            else:
                b = fft2c(b)  # 执行2D FFT（5D输入）
            x = mask*b  # 应用掩码
            x = c2r(x)  # 将复数转换回实数

    return x


# 定义SS_H函数，用于执行灵敏度图的Hermitian转置操作
def SS_H(z,csm):
    z = r2c(z)  # 将输入转换为复数
    csm = r2c(csm)  # 将灵敏度图转换为复数
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)  # 应用灵敏度图的共轭并求和
    z = z*csm  # 再次应用灵敏度图
    return c2r(z)  # 将结果转换回实数

# 定义S_H函数，用于执行灵敏度图的Hermitian转置操作（不包括最后一步乘以csm）
def S_H(z,csm):
    z = r2c(z)  # 将输入转换为复数
    csm = r2c(csm)  # 将灵敏度图转换为复数
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)  # 应用灵敏度图的共轭并求和
    return c2r(z)  # 将结果转换回实数

# 定义SS_H_hat函数，用于执行傅里叶域中的灵敏度图Hermitian转置操作
def SS_H_hat(z,csm):
    z = r2c(z)  # 将输入转换为复数
    z = ifft2c_2d(z)  # 执行2D IFFT
    csm = r2c(csm)  # 将灵敏度图转换为复数
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)  # 应用灵敏度图的共轭并求和
    z = z*csm  # 再次应用灵敏度图
    z = fft2c_2d(z)  # 执行2D FFT
    return c2r(z)  # 将结果转换回实数

# 定义S_H_hat函数，用于执行傅里叶域中的灵敏度图Hermitian转置操作（不包括最后一步乘以csm）
def S_H_hat(z,csm):
    z = r2c(z)  # 将输入转换为复数
    z = ifft2c_2d(z)  # 执行2D IFFT
    csm = r2c(csm)  # 将灵敏度图转换为复数
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)  # 应用灵敏度图的共轭并求和
    z = fft2c_2d(z)  # 执行2D FFT
    return c2r(z)  # 将结果转换回实数

# 定义ch_to_nb函数，用于将通道维度移动到批次维度
def ch_to_nb(z,filt=None):
    z = r2c(z)  # 将输入转换为复数
    if filt==None:
        z = torch.permute(z,(1,0,2,3))  # 交换通道和批次维度
    else:
        z = torch.permute(z,(1,0,2,3))/filt  # 交换通道和批次维度，并应用滤波器
    return c2r(z)  # 将结果转换回实数

# 定义Emat_xyt_complex函数，用于执行复数域的正向和反向傅里叶变换操作
def Emat_xyt_complex(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = b * mask  # 应用掩码
            if b.ndim == 4:
                x = ifft2c_2d(b)  # 执行2D IFFT（4D输入）
            else:
                x = ifft2c(b)  # 执行2D IFFT（5D输入）
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask  # 执行2D FFT并应用掩码（4D输入）
            else:
                x = fft2c(b) * mask  # 执行2D FFT并应用掩码（5D输入）
    else:
        if inv:
            x = b * mask  # 应用掩码
            if b.ndim == 4:
                x = ifft2c_2d(x)  # 执行2D IFFT（4D输入）
            else:
                x = ifft2c(x)  # 执行2D IFFT（5D输入）
            x = x*torch.conj(csm)  # 应用灵敏度图的共轭
            x = torch.sum(x, 1)  # 沿通道维度求和
            x = torch.unsqueeze(x, 1)  # 添加通道维度

        else:
            b = b*csm  # 应用灵敏度图
            if b.ndim == 4:
                b = fft2c_2d(b)  # 执行2D FFT（4D输入）
            else:
                b = fft2c(b)  # 执行2D FFT（5D输入）
            x = mask*b  # 应用掩码

    return x


# 定义r2c函数，用于将实数张量转换为复数张量 x:[1,30,320,320]
# def r2c(x):
#     if isinstance(x, np.ndarray):
#         x = torch.from_numpy(x)  # 将numpy.ndarray转换为Tensor
#     re, im = torch.chunk(x, 2, 1)  # 将输入沿通道维度分为实部和虚部
#     x = torch.complex(re, im)  # 创建复数张量
#     return x
'''
解释
1. 输入检查：首先检查输入是否为 numpy.ndarray，如果是，则将其转换为 torch.Tensor。
2. 分割实部和虚部：
re：取前15个通道作为实部。
im：取后15个通道作为虚部。
3. 创建复数张量：
使用 torch.zeros 创建一个复数张量，维度为 [1, 30, 320, 320]，数据类型为 torch.complex64。
将实部和虚部分别放入复数张量的相应通道中。
输出
经过这个修改后的 r2c 函数处理后，返回的复数张量的维度将保持为 [1, 30, 320, 320]，并且包含了实部和虚部的信息。
'''
def r2c(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)  # 将numpy.ndarray转换为Tensor
    
    # 假设输入的 x 是 [1, 30, 320, 320]
    # 将前15个通道作为实部，后15个通道作为虚部
    re = x[:, :15, :, :]  # 取前15个通道作为实部
    im = x[:, 15:, :, :]  # 取后15个通道作为虚部
    
    # 创建复数张量，保持通道数为30
    x = torch.zeros(x.shape, dtype=torch.complex64)  # 创建一个复数张量，维度为 [1, 30, 320, 320]
    x[:, :15, :, :] = re  # 将实部放入前15个通道
    x[:, 15:, :, :] = im  # 将虚部放入后15个通道
    
    return x


# # 定义c2r函数，用于将复数张量转换为实数张量
# def c2r(x):
#     x = torch.cat([torch.real(x), torch.imag(x)], 1)  # 将实部和虚部沿通道维度连接
#     return x
# 定义c2r函数，用于将复数张量转换为实数张量
def c2r(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)  # 将numpy.ndarray转换为Tensor
    x = torch.cat([torch.real(x), torch.imag(x)], 1)  # 将实部和虚部沿通道维度连接
    return x


# 定义sos函数，用于计算复数张量的平方和的平方根（Sum of Squares）
def sos(x):
    xr = np.real(x)  # 提取实部
    xi = np.imag(x)  # 提取虚部
    x = np.power(np.abs(xr),2)+np.power(np.abs(xi),2)  # 计算实部和虚部的平方和
    x = np.sum(x, 1)  # 沿通道维度求和
    x = np.power(x,0.5)  # 计算平方根
    return x


# 定义Abs函数，用于计算复数张量的绝对值
def Abs(x):
    x = r2c(x)  # 将输入转换为复数
    return torch.abs(x)  # 计算绝对值


# 定义l2mean函数，用于计算L2范数的平均值
def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))  # 计算绝对值的平方的平均值
    return result


# 定义TV函数，用于计算总变差（Total Variation）
def TV(x, norm='L1'):
    nb, nc, nx, ny = x.size()  # 获取输入张量的维度
    Dx = torch.cat([x[:, :, 1:nx, :], x[:, :, 0:1, :]], 2)  # 计算x方向的差分
    Dy = torch.cat([x[:, :, :, 1:ny], x[:, :, :, 0:1]], 3)  # 计算y方向的差分
    Dx = Dx - x  # 计算x方向的梯度
    Dy = Dy - x  # 计算y方向的梯度
    tv = 0
    if norm == 'L1':
        tv = torch.mean(torch.abs(Dx)) + torch.mean(torch.abs(Dy))  # 计算L1范数的总变差
    elif norm == 'L2':
        Dx = Dx * Dx
        Dy = Dy * Dy
        tv = torch.mean(Dx) + torch.mean(Dy)  # 计算L2范数的总变差
    return tv

# 定义stdnormalize函数，用于对复数张量进行标准化
def stdnormalize(x):
    x = r2c(x)  # 将输入转换为复数
    result = c2r(x)/torch.std(x)  # 除以标准差并转换回实数
    return result

# 定义to_null_space函数，用于将输入投影到零空间
def to_null_space(x,mask,csm):
    Aobj = Aclass(csm, mask, torch.tensor(.01).cuda())  # 创建Aclass对象
    Rhs = Emat_xyt(x, False, csm, mask)  # 执行正向操作
    Rhs = Emat_xyt(Rhs, True, csm, mask)  # 执行反向操作

    x_null = x - myCG(Aobj, Rhs, x, 5)  # 使用共轭梯度法求解并计算零空间投影

    #x = ifft2c_2d(fft2c_2d(r2c(x))*(1-mask))
    #chc = torch.conj(r2c(csm))*r2c(csm)+1e-5
    #x = torch.conj(r2c(csm))/chc*x
    #x = c2r(torch.sum(x, keepdim=True, dim=1))
    #x = stdnormalize(x)
    #x = torch.min(torch.abs(chc))*c2r(x)
    return x_null 


                


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    A^{T}A * X + \lamda *X
    """
    def __init__(self, csm, mask, lam):
        # 初始化Aclass对象
        self.pixels = mask.shape[0] * mask.shape[1]  # 计算像素总数
        self.mask = mask  # 存储掩码
        self.csm = csm  # 存储线圈灵敏度图
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.).float())  # 计算缩放因子
        self.lam = lam  # 存储正则化参数

    def myAtA(self, img):
        # 执行A^T * A操作
        x = Emat_xyt(img, False, self.csm, self.mask)  # 正向变换
        x = Emat_xyt(x, True, self.csm, self.mask)  # 反向变换
        
        return x + self.lam * img  # 返回结果加上正则化项


def myCG(A, Rhs, x0, it):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    #print('Rhs1', Rhs.shape, Rhs.dtype) #Rhs1.shape torch.Size([2, 256, 232])
    x0 = torch.zeros_like(Rhs)  # 初始化x0为零向量
    Rhs = r2c(Rhs) + A.lam * r2c(x0)  # 计算右侧向量
    
    x = r2c(x0)  # 将x0转换为复数形式
    i = 0  # 初始化迭代计数器
    r = Rhs - r2c(A.myAtA(x0))  # 计算初始残差
    p = r  # 初始化搜索方向
    rTr = torch.sum(torch.conj(r)*r).float()  # 计算残差的内积

    while i < it:  # 开始共轭梯度迭代
        Ap = r2c(A.myAtA(c2r(p)))  # 计算A*p
        alpha = rTr / torch.sum(torch.conj(p)*Ap).float()  # 计算步长
        alpha = torch.complex(alpha, torch.tensor(0.).float().cuda())  # 将alpha转换为复数
        x = x + alpha * p  # 更新x
        r = r - alpha * Ap  # 更新残差
        rTrNew = torch.sum(torch.conj(r)*r).float()  # 计算新的残差内积
        beta = rTrNew / rTr  # 计算beta
        beta = torch.complex(beta, torch.tensor(0.).float().cuda())  # 将beta转换为复数
        p = r + beta * p  # 更新搜索方向
        i = i + 1  # 增加迭代计数
        rTr = rTrNew  # 更新rTr

    return c2r(x)  # 返回结果的实部


def restore_checkpoint(ckpt_dir, state, device):
    # 从检查点恢复模型状态
    loaded_state = torch.load(ckpt_dir, map_location=device)  # 加载保存的状态
    state['optimizer'].load_state_dict(loaded_state['optimizer'])  # 恢复优化器状态
    state['model'].load_state_dict(loaded_state['model'], strict=False)  # 恢复模型状态
    state['ema'].load_state_dict(loaded_state['ema'])  # 恢复EMA状态
    state['step'] = loaded_state['step']  # 恢复步数
    
    return state  # 返回恢复后的状态


def save_checkpoint(ckpt_dir, state):
      # if not tf.io.gfile.exists(ckpt_dir):
    #     tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    #     logging.warning(f"No checkpoint found at {ckpt_dir}. "
    #                     f"Returned the same state as input")
    #     return state
    # else:
    # 保存检查点
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),  # 保存优化器状态
        'model': state['model'].state_dict(),  # 保存模型状态
        'ema': state['ema'].state_dict(),  # 保存EMA状态
        'step': state['step']  # 保存当前步数
    }
    torch.save(saved_state, ckpt_dir)  # 将状态保存到文件


def complex_kernel_forward(filter, i):
    # 将复数卷积核转换为实数形式
    filter = torch.squeeze(filter[i])  # 去除多余的维度
    filter_real = torch.real(filter)  # 提取实部
    filter_img = torch.imag(filter)  # 提取虚部
    kernel_real = torch.cat([filter_real, -filter_img], 1)  # 构造实部卷积核
    kernel_imag = torch.cat([filter_img, filter_real], 1)  # 构造虚部卷积核
    kernel_complex = torch.cat([kernel_real, kernel_imag], 0)  # 合并实部和虚部
    return kernel_complex


def conv2(x1, x2):
    # 执行2D卷积操作
    return F.conv2d(x1.float(), x2.float(), padding='same')


def ksp2float(ksp, i):
    # 将k空间数据转换为浮点数形式
    kdata = torch.squeeze(ksp[i])  # 去除多余的维度
    if len(kdata.shape) == 3:
        kdata = torch.unsqueeze(kdata, 0)  # 如果是3D数据，添加一个维度

    kdata_float = torch.cat([torch.real(kdata), torch.imag(kdata)], 1)  # 将实部和虚部拼接
    return kdata_float


def spirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc, nx, ny
    """
    nb = ksp.shape[0]  # 获取批次大小

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))  # 调整维度顺序
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)

    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))  # 调整维度顺序
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))  # 调整维度顺序

    re, im = torch.chunk(res_i, 2, 1)  # 分离实部和虚部
    res = torch.complex(re, im) - ksp  # 构造复数结果并减去原始k空间数据
    return res


def adjspirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc_s, nx, ny
    """

    nb = kernel.shape[0]  # 获取批次大小
    filter = torch.permute(kernel, (0, 2, 1, 3, 4))  # 调整卷积核维度顺序
    filter = torch.conj(filter.flip(dims=[-2, -1]))  # 计算卷积核的共轭并翻转

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))  # 调整k空间数据维度顺序
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)

    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))  # 调整结果维度顺序
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))  # 调整k空间数据维度顺序

    re, im = torch.chunk(res_i, 2, 1)  # 分离实部和虚部

    res = torch.complex(re, im) - ksp  # 构造复数结果并减去原始k空间数据

    return res


def dot_batch(x1, x2):
    # 计算批量点积
    batch = x1.shape[0]  # 获取批次大小
    res = torch.reshape(x1 * x2, (batch, -1))  # 将乘积重塑为2D张量
    return torch.sum(res, 1)  # 沿第二个维度求和


class ConjGrad:
    def __init__(self, A, rhs, max_iter=5, eps=1e-10):
        # 初始化共轭梯度求解器
        self.A = A  # 线性算子
        self.b = rhs  # 右侧向量
        self.max_iter = max_iter  # 最大迭代次数
        self.eps = eps  # 收敛阈值

    def forward(self, x):
        # 执行共轭梯度求解
        x = CG(x, self.b, self.A, max_iter=self.max_iter, eps=self.eps)
        return x

def CG(x, b, A, max_iter, eps):
    # 共轭梯度算法实现
    b = b + eps*x  # 添加正则化项
    r = b - A.A(x)  # 计算初始残差
    p = r  # 初始化搜索方向
    rTr = dot_batch(torch.conj(r), r)  # 计算残差的内积
    reshape = (-1,) + (1,) * (len(x.shape) - 1)  # 定义重塑形状
    num_iter = 0  # 初始化迭代计数器
    for iter in range(max_iter):
        if rTr.abs().max() < eps:  # 检查收敛条件
            break
        Ap = A.A(p)  # 计算A*p
        alpha = rTr / dot_batch(torch.conj(p), Ap)  # 计算步长
        alpha = torch.reshape(alpha, reshape)  # 重塑alpha
        x = x + alpha * p  # 更新x
        r = r - alpha * Ap  # 更新残差
        rTrNew = dot_batch(torch.conj(r), r)  # 计算新的残差内积
        beta = rTrNew / rTr  # 计算beta
        beta = torch.reshape(beta, reshape)  # 重塑beta
        p = r + beta * p  # 更新搜索方向
        rTr = rTrNew  # 更新rTr

        num_iter += 1  # 增加迭代计数
    return x  # 返回求解结果

def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''

    tmp = im2row(data, kernel_size)  # 将图像转换为行向量
    tsx, tsy, tsz = tmp.shape[:]  # 获取转换后的形状
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')  # 重塑为2D矩阵
    return np.dot(A.T.conj(), A)  # 计算A^H * A


def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]  # 获取输入图像的形状
    wx, wy = win_shape[:]  # 获取窗口大小
    sh = (sx-wx+1)*(sy-wy+1)  # 计算输出的行数
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)  # 初始化结果数组

    count = 0
    for y in range(wy):
        for x in range(wx):
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz))  # 提取并重塑图像块
            count += 1
    return res

def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):

    kx, ky = kernel_size[:]  # 获取核大小
    if sampling is None:
        sampling = np.ones((*kernel_size, ncoils))  # 如果没有提供采样，创建全1采样
    dummyK = np.zeros((kx, ky, ncoils))  # 创建虚拟核
    dummyK[int(kx/2), int(ky/2), coil] = 1  # 设置中心点为1

    idxY = np.where(dummyK)  # 获取非零元素的索引
    idxY_flat = np.sort(
        np.ravel_multi_index(idxY, dummyK.shape, order='F'))  # 将多维索引转换为一维索引
    sampling[idxY] = 0  # 在采样中将对应位置设为0
    idxA = np.where(sampling)  # 获取采样中非零元素的索引
    idxA_flat = np.sort(
        np.ravel_multi_index(idxA, sampling.shape, order='F'))  # 将多维索引转换为一维索引

    Aty = AtA[:, idxY_flat]  # 提取AtA的相关列
    Aty = Aty[idxA_flat]  # 提取Aty的相关行

    AtA0 = AtA[idxA_flat, :]  # 提取AtA的相关行
    AtA0 = AtA0[:, idxA_flat]  # 提取AtA0的相关列

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)  # 初始化核
    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda  # 计算正则化参数
    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0])*lamda, Aty)  # 求解线性方程

    kernel[idxA_flat] = rawkernel.squeeze()  # 将解填入核中
    kernel = np.reshape(kernel, sampling.shape, order='F')  # 重塑核

    return(kernel, rawkernel)


def spirit_calibrate(acs, kSize, lamda=0.001, filtering=False, verbose=True): # lamda=0.01
    nCoil = acs.shape[-1]  # 获取线圈数量
    AtA = dat2AtA(acs,kSize)  # 计算校准矩阵
    if filtering: # singular value threshing
        if verbose:
            ic('prefiltering w/ opth')
        U,s,Vh = svd(AtA, full_matrices=False)  # 进行奇异值分解
        k = optht(AtA, sv=s, sigma=None)  # 计算最优阈值
        if verbose:
            print('{}/{} kernels used'.format(k, len(s)))
        AtA= (U[:, :k] * s[:k] ).dot( Vh[:k,:])  # 重构AtA
    
   
    spirit_kernel = np.zeros((nCoil,nCoil,*kSize),dtype='complex128')  # 初始化SPIRIT核
    for c in range(nCoil): #tqdm(range(nCoil)):
        tmp, _ = calibrate_single_coil(AtA,kernel_size=kSize,ncoils=nCoil,coil=c,lamda=lamda)  # 校准单个线圈
        spirit_kernel[c] = np.transpose(tmp,[2,0,1])  # 调整维度顺序
        
    spirit_kernel = np.transpose(spirit_kernel,[2,3,1,0]) # Now same as matlab!
    GOP = np.transpose(spirit_kernel[::-1,::-1],[3,2,0,1])  # 计算GOP
    GOP = GOP.copy()
    for n in range(nCoil):
        GOP[n,n,kSize[0]//2,kSize[1]//2] = -1  # 设置GOP对角线元素
    return spirit_kernel


class Aclass_spirit:
    def __init__(self, kernel, mask, lam):
        # 初始化SPIRIT类
        self.kernel = kernel  # SPIRIT核
        self.mask = 1 - mask  # 采样掩码的补集
        self.lam = lam  # 正则化参数

    def ATA(self, ksp):
        # 执行A^T * A操作
        ksp = spirit(self.kernel, ksp)  # 正向SPIRIT变换
        ksp = adjspirit(self.kernel, ksp)  # 反向SPIRIT变换
        return ksp

    def A(self, ksp):
        # 执行A操作
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res


def ista_spirit(x0, b, kernel, mask, eta, thr, steps):
    # ISTA算法实现SPIRIT重建
    wave_name = 'haar'  # 小波类型
    mode = 'zero'       # 边界填充模式
    device = x0.device  # 获取设备
    dwt = wavelets.DWTForward(J=3, mode=mode, wave=wave_name).to(device)  # 创建正向小波变换
    idwt = wavelets.DWTInverse( mode=mode, wave=wave_name).to(device)  # 创建反向小波变换
    x = x0  # 初始化解
    for i in range(steps):
        grad = spirit(kernel, x)  # 计算梯度
        grad = adjspirit(kernel, grad)  # 计算梯度的伴随
        x = x - eta*grad  # 梯度下降步骤
        im = ifft2c_2d(x)  # 逆傅里叶变换
        # wavelet regularization
        Yl, Yh = dwt(c2r(im).float())  # 小波分解
        for h in range(3):
            Yh[h] = torch.sign(Yh[h])*torch.relu(torch.abs(Yh[h])-thr)  # 软阈值处理
        im = r2c(idwt((Yl,Yh)).float())  # 小波重构
        x = fft2c_2d(im)  # 傅里叶变换
        # projection
        x = (1-mask)*x + b  # 数据一致性投影
    return x


def sense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm), 1, keepdim=True)  # SENSE重建
    res = fft2c_2d(m * csm)  # 转回k空间
    return res - ksp  # 返回残差


def adjsense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm), 1, keepdim=True)  # SENSE重建
    res = fft2c_2d(m * csm)  # 转回k空间
    return res - ksp  # 返回残差


class Aclass_sense:
    def __init__(self, csm, mask, lam):
        # 初始化SENSE类
        self.s = csm  # 线圈灵敏度图
        self.mask = 1 - mask  # 采样掩码的补集
        self.lam = lam  # 正则化参数

    def ATA(self, ksp):
        # 执行A^T * A操作
        Ax = sense(self.s, ksp)  # 正向SENSE变换
        AHAx = adjsense(self.s, Ax)  # 反向SENSE变换
        return AHAx

    def A(self, ksp):
        # 执行A操作
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res


def cgSPIRiT(x0, ksp, kernel, mask, niter, lam):
    # 共轭梯度法求解SPIRIT重建
    Aobj = Aclass_spirit(kernel, mask, lam)  # 创建SPIRIT对象
    y = - (1 - mask) * Aobj.ATA(ksp)  # 计算右侧向量
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)  # 创建共轭梯度求解器
    x = cg_iter.forward(x=x0)  # 执行共轭梯度求解
    x =  x * (1 - mask) + ksp  # 数据一致性
    return x


class Aclass_spiritv2:
    def __init__(self, kernel, mask, lam1, lam2):
        # 初始化SPIRITv2类
        self.kernel = kernel  # SPIRIT核
        self.mask = mask  # 采样掩码
        self.lam1 = lam1  # 正则化参数1
        self.lam2 = lam2  # 正则化参数2

    def ATA(self, ksp):
        # 执行A^T * A操作
        ksp = spirit(self.kernel, ksp)  # 正向SPIRIT变换
        ksp = adjspirit(self.kernel, ksp)  # 反向SPIRIT变换
        return ksp

    def A(self, ksp):
        # 执行A操作
        res = self.lam1*self.ATA(ksp) + self.mask*ksp + self.lam2 * ksp
        return res

def cgSPIRiTv2(x0, ksp, kernel, mask, niter, lam1, lam2):
    # 共轭梯度法求解SPIRITv2重建
    Aobj = Aclass_spiritv2(kernel, mask, lam1, lam2)  # 创建SPIRITv2对象
    y = ksp  # 设置右侧向量
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)  # 创建共轭梯度求解器
    x = cg_iter.forward(x=x0)  # 执行共轭梯度求解
    return x

class Aclass_Self:
    def __init__(self, kernel, lam):
        # 初始化Self类
        self.kernel = kernel  # SPIRIT核
        self.lam = lam  # 正则化参数

    def ATA(self, ksp):
        # 执行A^T * A操作
        ksp = spirit(self.kernel, ksp)  # 正向SPIRIT变换
        ksp = adjspirit(self.kernel, ksp)  # 反向SPIRIT变换
        return ksp

    def A(self, ksp):
        # 执行A操作
        res = self.ATA(ksp) + self.lam * ksp
        return res

def cgSELF(x0, kernel, niter, lam):
    # 共轭梯度法求解Self重建
    Aobj = Aclass_Self(kernel, lam)  # 创建Self对象
    y = 0  # 设置右侧向量为0
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)  # 创建共轭梯度求解器
    x = cg_iter.forward(x=x0)  # 执行共轭梯度求解
    return x


def cgSENSE(x0, ksp, csm, mask, niter, lam):
    # 共轭梯度法求解SENSE重建
    Aobj = Aclass_sense(csm, mask, lam)  # 创建SENSE对象
    y = - (1 - mask) * Aobj.ATA(ksp)  # 计算右侧向量
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)  # 创建共轭梯度求解器
    x = cg_iter.forward(x=x0)  # 执行共轭梯度求解
    x = x * (1 - mask) + ksp  # 数据一致性
    res = torch.sum(ifft2c_2d(x) * torch.conj(csm), 1, keepdim=True)  # 计算最终重建结果
    return x, res
'''
`cgSENSE` 和 `sense` 方法的主要区别在于它们的功能和实现方式。以下是对这两个方法的详细比较：

### 1. 功能

- **`sense` 方法**：
  - 该方法用于执行 SENSE 重建。它接受线圈灵敏度图（csm）和 k-space 数据（ksp）作为输入，计算图像域的重建结果，并将其转换回 k-space。
  - 返回的是 k-space 数据的残差，即重建结果与原始 k-space 数据之间的差异。

- **`cgSENSE` 方法**：
  - 该方法实现了基于共轭梯度法的 SENSE 重建。它使用 `Aclass_sense` 类来定义正向和反向操作，并通过迭代优化来求解重建问题。
  - 返回的是重建后的图像和最终的重建结果（res），即通过 SENSE 重建得到的图像在 k-space 中的表示。

### 2. 实现方式

- **`sense` 方法**：
  - 直接计算 SENSE 重建，使用 `ifft2c_2d` 和 `fft2c_2d` 函数进行傅里叶变换。
  - 计算过程相对简单，主要是通过线圈灵敏度图对 k-space 数据进行加权和求和。

```python
def sense(csm, ksp):
    m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm), 1, keepdim=True)  # SENSE重建
    res = fft2c_2d(m * csm)  # 转回k空间
    return res - ksp  # 返回残差
```

- **`cgSENSE` 方法**：
  - 使用共轭梯度法进行迭代优化，首先计算右侧向量 `y`，然后通过 `ConjGrad` 类进行求解。
  - 该方法更复杂，适用于需要迭代优化的场景，能够在一定程度上提高重建的精度。

```python
def cgSENSE(x0, ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)  # 创建SENSE对象
    y = - (1 - mask) * Aobj.ATA(ksp)  # 计算右侧向量
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)  # 创建共轭梯度求解器
    x = cg_iter.forward(x=x0)  # 执行共轭梯度求解
    x = x * (1 - mask) + ksp  # 数据一致性
    res = torch.sum(ifft2c_2d(x) * torch.conj(csm), 1, keepdim=True)  # 计算最终重建结果
    return x, res
```

### 3. 输入和输出

- **`sense` 方法**：
  - 输入：线圈灵敏度图（csm）、k-space 数据（ksp）。
  - 输出：k-space 数据的残差。

- **`cgSENSE` 方法**：
  - 输入：初始图像（x0）、k-space 数据（ksp）、线圈灵敏度图（csm）、掩码（mask）、迭代次数（niter）、正则化参数（lam）。
  - 输出：重建后的图像和最终的重建结果（res）。

### 总结

- `sense` 方法是一个直接的 SENSE 重建实现，适合简单的重建任务。
- `cgSENSE` 方法则是一个基于共轭梯度法的迭代优化实现，适合需要更高精度的重建任务。
'''


def SPIRiT_Aobj(kernel,ksp):
    # SPIRIT变换
    ksp = spirit(kernel, ksp)  # 正向SPIRIT变换
    ksp = adjspirit(kernel, ksp)  # 反向SPIRIT变换
    return ksp


def add_noise(x, snr):
    # 添加噪声
    x_ = x.view(x.shape[0], -1)  # 将输入展平
    x_power = torch.sum(torch.pow(torch.abs(x_), 2), dim=1, keepdim=True) / x_.shape[1]  # 计算信号功率
    snr = 10 ** (snr / 10)  # 将dB转换为线性比例
    noise_power = x_power / snr  # 计算噪声功率
    reshape = (-1,) + (1,) * (len(x.shape) - 1)  # 定义重塑形状
    noise_power = torch.reshape(noise_power, reshape)  # 重塑噪声功率
    if x.dtype == torch.float32:
        noise = torch.sqrt(noise_power) * torch.randn(x.size(), device=x.device)  # 生成实数噪声
    else:
        noise = torch.sqrt(0.5 * noise_power) * (torch.complex(torch.randn(x.size(), device=x.device),
                                                               torch.randn(x.size(), device=x.device)))  # 生成复数噪声
    return x + noise  # 返回添加噪声后的信号


def blur_and_noise(x, kernel_size=7, sig=0.1, snr=10):
    # 添加模糊和噪声
    x_org = x  # 保存原始输入
    transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sig)  # 创建高斯模糊变换
    if x.dtype == torch.float32:
        x_ = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))  # 重塑实数输入
    else:
        x = c2r(x)  # 将复数转换为实数
        x_ = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))  # 重塑复数输入

    x_blur = transform(x_)  # 应用高斯模糊
    x_blur = torch.reshape(x_blur, x.shape)  # 重塑回原始形状
    x_blur_noise = add_noise(x_blur, snr=snr)  # 添加噪声
    if x_org.dtype == torch.float32:
        return x_blur_noise  # 返回实数结果
    else:
        return r2c(x_blur_noise)  # 返回复数结果

def matmul_cplx(x1, x2):
    # 定义复数矩阵乘法函数
    # 使用torch.view_as_complex将实部和虚部堆叠的结果转换为复数
    return torch.view_as_complex(
        # 使用torch.stack堆叠实部和虚部的计算结果
        torch.stack((
            # 计算复数乘法的实部：(a+bi)(c+di) = (ac-bd) + (ad+bc)i 中的 ac-bd
            x1.real @ x2.real - x1.imag @ x2.imag, 
            # 计算复数乘法的虚部：(a+bi)(c+di) = (ac-bd) + (ad+bc)i 中的 ad+bc
            x1.real @ x2.imag + x1.imag @ x2.real
        ), dim=-1))

def Gaussian_mask(nx, ny, Rmax, t, Fourier=True):
    # 定义生成高斯掩码的函数
    # 处理x方向的索引
    if nx % 2 == 0:
        # 如果nx是偶数，生成从-nx/2到nx/2-1的整数序列
        ix = np.arange(-nx//2, nx//2)
    else:
        # 如果nx是奇数，生成从-nx/2到nx/2的整数序列
        ix = np.arange(-nx//2, nx//2 + 1)

    # 处理y方向的索引，逻辑同上
    if ny % 2 == 0:
        iy = np.arange(-ny//2, ny//2)
    else:
        iy = np.arange(-ny//2, ny//2 + 1)

    # 计算x方向的频率权重
    wx = Rmax * ix / (nx / 2)
    # 计算y方向的频率权重
    wy = Rmax * iy / (ny / 2)

    # 使用meshgrid生成二维网格
    rwx, rwy = np.meshgrid(wx, wy)
    
    # 根据Fourier参数选择不同的高斯函数形式
    if Fourier:
        # 傅里叶域的高斯函数
        R = np.exp(-((rwx ** 2 + rwy ** 2)* t ** 2) / 2 )
    else:
        # 空间域的高斯函数
        R = np.exp(-(rwx ** 2 + rwy ** 2) / (2 * t ** 2))
        
    # 将结果转换为32位浮点数
    W = R.astype(np.float32)

    # 返回生成的高斯掩码
    return W

'''
orgk、atb 和 minv 是三个重要的变量，它们分别存储了欠采样的 k-space 数据、反投影数据和归一化因子。以下是对这三个变量的详细解释：
变量解释
orgk：
含义：orgk 存储了欠采样的 k-space 数据。k-space 是 MRI 数据的频域表示，包含了图像的频率信息。
生成方式：通过应用正向操作符 A（即 usp 函数）对原始图像数据 org 进行欠采样得到。
atb：
含义：atb 存储了反投影数据。反投影数据是通过应用反向操作符 At（即 usph 函数）对欠采样的 k-space 数据进行处理得到的。
生成方式：通过应用反向操作符 At 对欠采样的 k-space 数据 y 进行处理得到。
minv：
含义：minv 存储了每个切片的归一化因子。归一化因子用于对欠采样的 k-space 数据进行归一化处理。
生成方式：通过应用反向操作符 At 对欠采样的 k-space 数据 y 进行处理时得到。

详细解释
orgk：
初始化：orgk = np.empty(org.shape, dtype=np.complex64) 创建了一个与输入数据 org 形状相同的复数数组。
生成：在循环中，通过 A(org[i]) 得到欠采样的 k-space 数据 orgk[i]。
atb：
初始化：atb = np.empty(org.shape, dtype=np.complex64) 创建了一个与输入数据 org 形状相同的复数数组。
生成：在循环中，通过 At(y) 得到反投影数据 atb[i]。
minv：
初始化：minv = np.zeros((nSlice,), dtype=np.complex64) 创建了一个一维复数数组，用于存储每个切片的归一化因子。
生成：在循环中，通过 At(y) 得到归一化因子 minv[i]。
总结
orgk：存储了欠采样的 k-space 数据，通过正向操作符 A 生成。
atb：存储了反投影数据，通过反向操作符 At 生成。
minv：存储了每个切片的归一化因子，通过反向操作符 At 生成。
'''
def generateUndersampled(org,mask):
    # 获取输入数据的形状
    nSlice,nch,nrow,ncol=org.shape
    # 创建与输入数据相同形状的复数数组，用于存储欠采样的k空间数据
    orgk=np.empty(org.shape,dtype=np.complex64)
    # 创建与输入数据相同形状的复数数组，用于存储反投影数据
    atb=np.empty(org.shape,dtype=np.complex64)
    # 创建一个一维复数数组，用于存储每个切片的归一化因子
    minv=np.zeros((nSlice,),dtype=np.complex64)
    # 对每个切片进行处理
    for i in range(nSlice):
        # 定义正向操作符A
        A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
        # 定义反向操作符At
        At = lambda z: usph(z,mask[i],nch,nrow,ncol)
        # 应用正向操作符，获取欠采样的k空间数据
        orgk[i],y=A(org[i])
        # 应用反向操作符，获取反投影数据和归一化因子
        atb[i],minv[i]=At(y)
        # 对欠采样的k空间数据进行归一化
        orgk[i]=orgk[i]/minv[i]
    # 删除原始数据，释放内存
    del org
    # 返回欠采样的k空间数据、反投影数据和归一化因子
    return orgk,atb,minv

def usp(x,mask,nch,nrow,ncol):
    """ 这是论文中定义的A操作符 This is a the A operator as defined in the paper"""
    # 将输入数据重塑为(nch,nrow,ncol)的形状
    kspace=np.reshape(x,(nch,nrow,ncol))
    # 根据掩码选择非零元素，实现欠采样
    res=kspace[mask!=0]
    # 返回重塑后的k空间数据和欠采样结果
    return kspace,res

def usph(kspaceUnder,mask,nch,nrow,ncol):
    """ 这是论文中定义的A^T操作符 This is a the A^T operator as defined in the paper"""
    # 创建一个与原始k空间数据相同形状的零数组
    temp=np.zeros((nch,nrow,ncol),dtype=np.complex64)
    # 将欠采样数据填充到非零掩码位置
    temp[mask!=0]=kspaceUnder
    # 计算数据的标准差作为归一化因子
    minv=np.std(temp)
    # 对数据进行归一化
    temp=temp/minv
    # 返回归一化后的数据和归一化因子
    return temp,minv

# # 主要用于进行SPIRiT校准，并使用校准后的SPIRiT核进行图像重建。
# # 这是一个包含校准区域数据的变量，它是一个 NumPy 数组
# '''
# calib: 这是一个包含校准区域数据的变量，它是一个 NumPy 数组;
# 作用是将 calib 变量裁剪成一个 32x32 的图像，并将其维度顺序调整为 (通道数, 高度, 宽度)。
# spirit_crop: 这个函数用于裁剪图像，它接受一个图像、裁剪区域的宽度和高度以及目标图像的宽度作为参数。
# np.transpose(calib.squeeze().cpu().numpy(), (1, 2, 0)): 这部分代码将 calib 变量从 PyTorch 张量转换为 NumPy 数组，并调整其维度顺序。
# r2c(atb).shape[1]: 这部分代码获取 atb 变量的第二维度的长度，用于确定目标图像的宽度。
# '''
# calib = spirit_crop(np.transpose(calib.squeeze().cpu().numpy(), (1, 2, 0)), 32, 32, r2c(atb).shape[1])
# '''
# spirit_calibrate: 这个函数用于计算 SPIRiT 核，它接受校准区域数据、核的大小、正则化参数、是否进行滤波以及是否打印日志信息作为参数。
# calib: 这是前面裁剪后的校准区域数据。
# (5, 5): 这是 SPIRiT 核的大小。
# lamda=0.25: 这是正则化参数，用于控制正则化的强度。
# filtering=False: 表示不进行滤波。
# verbose=True: 表示打印日志信息。
# 这一行代码的作用是使用 calib 变量计算一个 5x5 的 SPIRiT 核
# '''
# kernel = spirit_calibrate(calib, (5, 5), lamda=0.25, filtering=False, verbose=True)
# # to(torch.float32).to(device)
# '''
# torch.from_numpy(kernel): 将 NumPy 数组 kernel 转换为 PyTorch 张量。
# torch.permute(..., (3, 2, 0, 1)): 调整张量的维度顺序。
# unsqueeze(0): 在张量的第一个维度上添加一个维度。
# 这一行代码的作用是将 kernel 变量转换为 PyTorch 张量，并调整其维度顺序，使其符合 SPIRiT 核的格式。
# '''
# kernel = torch.permute(torch.from_numpy(kernel), (3, 2, 0, 1)).unsqueeze(0)
# '''
# c2r(kernel).to(torch.float32): 将 kernel 变量转换为实数张量，并将其数据类型转换为 torch.float32。
# r2c(...): 将实数张量转换为复数张量。
# to(device): 将张量移动到指定的设备上。
# '''
# kernel = r2c(c2r(kernel).to(torch.float32)).to(device)
# print('kernel', kernel.shape, kernel.dtype, r2c(atb).shape, r2c(atb).dtype, mask.shape, mask.dtype)
# '''
# cgSPIRiT: 这个函数用于使用 SPIRiT 核进行图像重建，它接受一个初始图像、k 空间数据、SPIRiT 核、掩码、迭代次数和正则化参数作为参数。
# r2c(atb).to(torch.complex64): 将 atb 变量转换为复数张量，并将其数据类型转换为 torch.complex64。
# kernel: 这是前面计算的 SPIRiT 核。
# mask: 这是 k 空间数据的掩码，用于指示哪些数据点是已知的。
# 30: 这是迭代次数。
# 1e-7: 这是正则化参数。
# 这一行代码的作用是使用 kernel 变量和 mask 变量对 atb 变量进行图像重建，并返回重建后的图像。
# '''
# x0 = cgSPIRiT(r2c(atb).to(torch.complex64), r2c(atb).to(torch.complex64), kernel, mask, 30, 1e-7).to(torch.complex64)
# '''
# ACS:
# 在核磁共振成像（MRI）中，ACS（Auto-Calibration Signal）区域是指用于自校准信号的区域。
# 这是采集 k 空间数据的一个子集，用于校准或估计重建过程中所需的感应线圈灵敏度信息。
# ACS 区域通常位于 k 空间的中心，因为那里包含了低频信息，有助于提高图像的信噪比和重建质量。

# atb
# 在核磁共振成像（MRI）中，atb 通常指代图像重建过程中的一个变量或数据。
# 具体来说，atb 可能是经过某种预处理或变换后的 k 空间数据，用于进一步的图像重建步骤。
# 它可能涉及信号的加权、滤波或从 k 空间到图像空间的变换。具体内容会根据上下文和使用的算法而有所不同。

# CSM:
# '''

'''
相关问题：
这段代码中`sense`和`adjsense`函数的实现细节有何不同，以及这种差异对最终重建结果的影响是什么?
共轭梯度法(CG)在该SENSE重建算法中扮演什么角色?如何调整CG算法的参数(`max_iter`，｀eps`)以平衡重建速度和精度?代码中使用了多种归一化方法(`normalize`，`normalize_complex`，`normalize_12')，这些方法分别适用于什么场景，它们之间有何区别?
Emat xyt`函数是如何实现空间和时间维度的处理的?不同参数设置下，该函数的计算复杂度如何变化?
'''