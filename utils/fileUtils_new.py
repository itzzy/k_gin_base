import numpy as np
from PIL import Image
import scipy.io as sio
import numpy as np
import os
import h5py
# from /data0/zhiyong/code/github/itzzy_git/k-gin_kv/UTILS.py import FFT2c IFFT2c 

# import sys
# sys.path.append('/data0/zhiyong/code/github/itzzy_git/k-gin_kv/UTILS.py')
# from UTILS.py import FFT2c,IFFT2c
# from ..UTILS import FFT2C, IFFT2C
from fastmriBaseUtils import FFT2c,IFFT2c
# 这种方法是将上一级目录添加到模块搜索路径中，然后进行导入。
# import sys
# sys.path.append("..")
# from UTILS import FFT2c, IFFT2c
# from UTILS import FFT2C, IFFT2C
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_mat(fn_im_path):
    try:
        f = sio.loadmat(fn_im_path)
    except Exception:
        try:
            f = h5py.File(fn_im_path, 'r')
        except IOError:
            # print("File {} is defective and cannot be read!".format(fn_im_path))
            raise IOError("File {} is defective and cannot be read!".format(fn_im_path))
    return f

### 原始kspace和csm保存为图片（label）
# 加载 k-space 数据和线圈灵敏度图
#  train_subjs: '/nfs/zzy/data/k_gin_data/k_cine_multicoil_training.npy' # a demo
#   train_maps: '/nfs/zzy/data/k_gin_data/csm_cine_multicoil_training.npy' # a demo
#   val_subjs: '/nfs/zzy/data/k_gin_data/k_cine_multicoil_test.npy' # a demo
#   val_maps: '/nfs/zzy/data/k_gin_data/csm_cine_multicoil_test.npy' # a demo
# data = np.load('/nfs/zzy/data/k_gin_data/k_cine_multicoil_test.npy')
# csm = np.load('/nfs/zzy/data/k_gin_data/csm_cine_multicoil_test.npy')
# print("data:", data.shape)  # data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)

# # 只取第一个样本
# data = data[0, :, :, :, :]  # (coil=20, 18, 192, 192)
# csm = csm[0, :, :, :, :]    # (coil=20, 18, 192, 192)

# # 对 k-space 数据进行逆傅里叶变换，并与 CSM 结合生成图像
# img = np.sum(IFFT2c(data) * np.conj(csm), axis=0)  # (18, 192, 192)
# print("img:", img.shape)  # img: (18, 192, 192)

# # 归一化并调整亮度
# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# # 创建 base 目录
# output_dir = 'base'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 保存前五张图片
# for i in range(5):
#     plt.imshow(img_brightened[i], cmap='gray')
#     plt.title(f'Frame {i}')
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f'frame_{i}.png'))  # 保存为 PNG 文件
#     plt.close()  # 关闭当前图像，避免重叠

# print(f"前五张图片已保存到 {output_dir} 目录下。")


######原kspace和mask做欠采样
# 加载 k-space 数据和线圈灵敏度图
# data = np.load('/nfs/zzy/data/k_gin_data/k_cine_multicoil_test.npy')
# csm = np.load('/nfs/zzy/data/k_gin_data/csm_cine_multicoil_test.npy')
# print("data-1:", data.shape)  # data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192) data: (118, 20, 18, 192, 192)

# # 只取第一个样本
# data = data[0, :, :, :, :]  # (coil=20, 18, 192, 192)
# print("data-2:", data.shape)
# csm = csm[0, :, :, :, :]    # (coil=20, 18, 192, 192)

# # 加载欠采样掩码
# C = load_mat('/nfs/zzy/code/k_gin_base/masks/VISTA/e_192x18_acs4_R4.mat')
# mask = C['mask'][:]  # mask-type-1 (192, 18)
# print('mask-type-1', mask.shape)  # mask-type mask-type-1 (192, 18)
# mask = np.transpose(mask, [1, 0])  # 转置为 (18, 192)
# print('mask-type-2', mask.shape)  # mask-type (18, 192)



# # 扩展掩码维度以匹配 k-space 数据
# mask = np.expand_dims(mask, axis=1)  # mask (18, 1, 192)
# print('mask', mask.shape)  # mask: mask (18, 1, 192)

# # 对 k-space 数据进行欠采样
# data = data * mask  # (20, 192, 18, 192)
# print("data-3:", data.shape) #data-3: (20, 18, 192, 192)


# # 对欠采样的 k-space 数据进行逆傅里叶变换，并与 CSM 结合生成图像
# img = np.sum(IFFT2c(data) * np.conj(csm), axis=0)  # (18, 192, 192)
# print("img:", img.shape)  # img: (18, 192, 192)

# # 归一化并调整亮度
# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# # 创建 testzerofilling 目录
# output_dir = 'testzerofilling'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# # 保存前五张图片
# for i in range(5):
#     break
#     plt.imshow(img_brightened[i], cmap='gray')
#     plt.title(f'Frame {i}')
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f'frame_{i}.png'))  # 保存为 PNG 文件
#     plt.close()  # 关闭当前图像，避免重叠

# # 生成 GIF 动画
# def animate(frame):
#     plt.imshow(img_brightened[frame], cmap='gray')
#     plt.title(f'Frame {frame}')
#     plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# anim.save(os.path.join(output_dir, 'testzerofilling.gif'), writer='imagemagick')  # 保存 GIF 文件
# print(f"前五张图片和 GIF 已保存到 {output_dir} 目录下。")

# mask_bak = mask
# mask_bak = np.expand_dims(mask_bak, axis=0)  # 添加 batch 维度
# print('mask_bak-type-1', mask_bak.shape)  # mask_bak-type-1 (1, 192, 18)
# mask_bak = np.expand_dims(mask_bak, axis=0)  # 添加 time 维度
# print('mask_bak-type-2', mask_bak.shape)  # mask_bak-type-2 (1, 1, 192, 18)
# mask_bak = np.tile(mask_bak, (1, 192, 1, 1))  # 广播到完整形状
# print('mask_bak-type-3', mask_bak.shape)  # mask_bak-type-3 (1, 192, 192, 18)
# # 调整维度顺序
# mask_bak = np.transpose(mask_bak, (0, 3, 1, 2))  # 将第 3 维度移到第 1 维度
# print('mask_bak new shape:', mask_bak.shape)  # 输出: (1, 18, 192, 192)
# data_bak = data * mask_bak  # (20, 192, 18, 192)
# print("data_bak-1:", data_bak.shape) #data-3: (20, 18, 192, 192)
# # 对欠采样的 k-space 数据进行逆傅里叶变换，并与 CSM 结合生成图像
# img = np.sum(IFFT2c(data_bak) * np.conj(csm), axis=0)  # (18, 192, 192)
# print("img:", img.shape)  # img: (18, 192, 192)

# # 归一化并调整亮度
# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# # 创建 testzerofilling 目录  此mask可能不对
# output_dir = 'testzerofilling-bak'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# # 保存前五张图片
# for i in range(5):
#     plt.imshow(img_brightened[i], cmap='gray')
#     plt.title(f'Frame {i}')
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f'frame_{i}.png'))  # 保存为 PNG 文件
#     plt.close()  # 关闭当前图像，避免重叠

# # 生成 GIF 动画
# def animate(frame):
#     plt.imshow(img_brightened[frame], cmap='gray')
#     plt.title(f'Frame {frame}')
#     plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# anim.save(os.path.join(output_dir, 'testzerofilling-bak.gif'), writer='imagemagick')  # 保存 GIF 文件
# print(f"前五张图片和 GIF 已保存到 {output_dir} 目录下。")






# data = np.load('/nfs/zzy/data/k_gin_data/k_cine_multicoil_test.npy')
# csm = np.load('/nfs/zzy/data/k_gin_data/csm_cine_multicoil_test.npy')
# print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[0,:,:,:,:]
# csm = csm[0,:,:,:,:] 

# C =load_mat('/nfs/zzy/code/k_gin_base/masks/VISTA/e_192x18_acs4_R4.mat')
# mask = C['mask'][:]
# mask = np.transpose(mask,[1,0])
# print('mask-type',mask.shape)
# mask = np.expand_dims(mask, axis=1)
# print('mask',mask.shape)

# data = data*mask
# img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
# print("img:", img.shape)
# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# def animate(frame):
#    plt.imshow(img_brightened[frame], cmap='gray')  
#    plt.title('Frame {}'.format(frame))
#    plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# anim.save('testzerofilling.gif', writer='imagemagick')

# data = np.load('/data0/chentao/data/LplusSNet/data/20coil/k_cine_multicoil_test.npy')
# csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
# print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[0,:,:,:,:]
# csm = csm[0,:,:,:,:] 
# img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
# print("img:", img.shape)

# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# def animate(frame):
#    plt.imshow(img_brightened[frame], cmap='gray')  
#    plt.title('Frame {}'.format(frame))
#    plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# anim.save('test.gif', writer='imagemagick')


###### 保存为图片
# # 加载数据
# # data = np.load('/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy') #r4
# data = np.load('/nfs/zzy/code/k_gin_base/output/r8/out_1209r_8.npy') #r6
# print("data:", data.shape)  # data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)

# # 只取第一帧数据
# data = data[0:1, :, :, :]

# # 对数据进行逆傅里叶变换
# img = IFFT2c(data)
# img = img[0]  # 取第一个样本
# print("img:", img.shape)  # img: (18, 192, 192)

# # 归一化并调整亮度
# img_max = np.max(np.abs(img))
# img_norm = np.abs(img) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# # # 创建 r4 目录
# # output_dir = 'r4'
# # # 创建 r6 目录
# # output_dir = 'r6'
# # 创建 r8 目录
# output_dir = 'r8'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 保存前五张图片
# for i in range(5):
#     plt.imshow(img_brightened[i], cmap='gray')
#     plt.title(f'Frame {i}')
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f'frame_{i}.png'))  # 保存为 PNG 文件
#     plt.close()  # 关闭当前图像，避免重叠

# print(f"前五张图片已保存到 {output_dir} 目录下。")

'''
data = np.load('/data0/zhiyong/code/github/k-gin/out_1122.npy')
#csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
# data: (118, 18, 192, 192)
# img: (18, 192, 192)
print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[100:101,:,:,:]
data = data[80:81,:,:,:]
img = IFFT2c(data)
img = img[0]
print("img:", img.shape)

img_max = np.max(np.abs(img))
if img_max == 0 or np.isnan(img_max):
    img_norm = np.abs(img)  # 或者选择其他合理的处理方式
else:
    img_norm = np.abs(img) / img_max
    
brightness_factor = 3
img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

def animate(frame):
   plt.imshow(img_brightened[frame], cmap='gray')  
   plt.title('Frame {}'.format(frame))
   plt.axis('off')

anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# kv  zzy
anim.save('output_kgin_1122_80.gif', writer='imagemagick')
'''
# data = np.load('/data0/huayu/Aluochen/Mypaper5/k-gin_kv/out.npy')
# data = np.load('/data0/zhiyong/code/github/itzzy_git/k-gin-git/out_1118.npy')
# /data0/zhiyong/code/github/k-gin/out_1121.npy
# data = np.load('/data0/zhiyong/code/github/itzzy_git/k-gin-git/out_1119.npy')
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1121.npy')
# data = np.load('/data0/zhiyong/code/github/itzzy_git/k-gin-git/out.npy')
# out_1121_2.npy
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1122.npy')
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1127.npy')
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1127.npy')
# data = np.load('/data0/zhiyong/code/github/itzzy_git/k-gin_kv/out.npy')
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1121.npy')
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1130.npy')
# data = np.load('/data0/zhiyong/code/github/itzzy_git/k-gin_kv/out_1130_3.npy')
# /data0/zhiyong/code/github/k-gin/out_1201_2.npy

# /data0/zhiyong/code/github/k-gin/out_1201_2.npy
# /nfs/zzy/code/k_gin_base/output/r8/out_1209r_8.npy
# data = np.load('/data0/zhiyong/code/github/k-gin/out_1201_2.npy')
# data = np.load('/nfs/zzy/code/k_gin_base/output/r8/out_1209r_8.npy')
# data = np.load('/nfs/zzy/code/k_gin_base/output/r6/out_1206_1.npy')
# data = np.load('/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy')



#csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
# csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')

# data: (118, 18, 192, 192)
# img: (18, 192, 192)
# print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[100:101,:,:,:]
# data = data[100:101,:,:,:]
#csm = csm[100,:,:,:,:]
# csm = csm[100:101,:,:,:] 
# print('csm:',csm.shape)

#img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
# img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #

# img = IFFT2c(data)
# img = img[0]
# print("img:", img.shape)

# img_max = np.max(np.abs(img))
# if img_max == 0 or np.isnan(img_max):
#     img_norm = np.abs(img)  # 或者选择其他合理的处理方式
# else:
#     img_norm = np.abs(img) / img_max

# img_norm = np.abs(img) / img_max
# 
# 假设 img 是一个 numpy 数组，代表您的图像数据
# img_max = np.max(np.abs(img))  # 计算图像数据绝对值的最大值

# # 检查 img_max 是否为零，避免除以零的错误
# if img_max != 0:
#     img_norm = np.abs(img) / img_max  # 如果 img_max 不为零，则执行归一化操作
# else:
#     img_norm = np.zeros_like(img)  # 如果 img_max 为零，创建一个与 img 形状相同的全零数组

# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# def animate(frame):
#    plt.imshow(img_brightened[frame], cmap='gray')  
#    plt.title('Frame {}'.format(frame))
#    plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# kv  zzy
# anim.save('output-kgin-1127.gif', writer='imagemagick')
# anim.save('output-kgin-test-1130.gif', writer='imagemagick')
# k-gin_kv/out_1130_3
# anim.save('output-kgin_kv-out_1130_3.gif', writer='imagemagick')

# /data0/zhiyong/code/github/k-gin/out_1201.npy
# anim.save('output-kgin-out_1201_2.gif', writer='imagemagick')
# /data0/zhiyong/code/github/k-gin/out_1201_2.npy
# anim.save('output-kgin-out_1201_3.gif', writer='imagemagick')
# 1209r_8
# anim.save('output-kgin_1209r_8.gif', writer='imagemagick')
# anim.save('output-kgin_1206r_6.gif', writer='imagemagick')
# /nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy
# anim.save('output-kgin_1220_r4.gif', writer='imagemagick')







# data = np.load('/data0/chentao/data/LplusSNet/data/20coil/k_cine_multicoil_test.npy')
# data = np.load('/data0/zhiyong/code/github/itzzy_git/k-gin_kv/out_1118.npy')
# csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
# # data: (118, 18, 192, 192)
# print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[100,:,:,:]
# # data = data[100,:,:,:]
# csm = csm[100,:,:,:,:] 

# # C =loadmat('/data0/huayu/Aluochen/Mypaper5/e_192x18_acs4_R4.mat')
# C = load_mat('/data0/huayu/Aluochen/Mypaper5/e_192x18_acs4_R4.mat')
# mask = C['mask'][:]
# mask = np.transpose(mask,[1,0])
# mask = np.expand_dims(mask, axis=1)
# # mask (18, 1, 192)
# print('mask',mask.shape)

# # data = data*mask
# # data-mask: (18, 192, 192)
# print('data-mask:',data.shape)
# # img = np.expand_dims(np.sum(IFFT2c(data) * np.conj(csm), axis=0), axis=0) #
# # print("img:", img.shape)
# # ksp = FFT2c(img)

# #for i in range(ksp.shape[3]):
# #    
# #    for j in range(mask.shape[3]):
# # ksp[:, 10:15, :, 96:100] = ksp[:, 12:13, :, 96:100]
# # img = np.expand_dims(np.sum(ksp[:,10:15,:,:], axis=1), axis=1) / np.expand_dims(np.sum(mask[10:15,:,:], axis=0), axis=0)
# # #img = np.expand_dims(np.sum(ksp, axis=1), axis=1) / np.expand_dims(np.sum(mask, axis=0), axis=0)
# # #img = np.mean(ksp, axis=1, keepdims=True)
# # print("img:", img.shape)
# # #img = np.repeat(img, repeats=2, axis=1)

# # img = IFFT2c(img)
# # img = img[0] 

# img_max = np.max(np.abs(data))
# img_norm = np.abs(data) / img_max
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# def animate(frame):
#     plt.imshow(img_brightened[frame], cmap='gray')  
#     plt.title('Frame {}'.format(frame))
#     plt.axis('off')

# anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
# anim.save('test_k_cine_1120.gif', writer='imagemagick')


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from numpy.fft import ifft2 as IFFT2c


# def main(filepath,outpath):
#     try:
#         # 加载数据文件，首先打印出数据的原始形状
#         # data = np.load('/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy')
#         data = np.load(filepath)
#         print("Original data shape:", data.shape)

#         # 根据实际数据形状修改切片操作
#         # 这里暂时假设你需要对数据进行不同的切片操作，需要根据实际情况修改
#         data = data[100:101, :, :, :]
#         print("Sliced data shape:", data.shape)

#         # 傅里叶逆变换
#         img = IFFT2c(data)
#         img = img[0]
#         print("img:", img.shape)

#         # 归一化处理
#         img_max = np.max(np.abs(img))
#         if img_max == 0 or np.isnan(img_max):
#             img_norm = np.abs(img)
#         else:
#             img_norm = np.abs(img) / img_max

#         # 增加图像亮度
#         brightness_factor = 3
#         img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

#         def animate(frame):
#             plt.imshow(img_brightened[frame], cmap='gray')
#             plt.title('Frame {}'.format(frame))
#             plt.axis('off')

#         # 创建动画
#         anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
#         # anim.save('output-kgin_1220_r4.gif', writer='imagemagick')
#         anim.save(outpath, writer='imagemagick')
#     except Exception as e:
#         print(f"An error occurred: {e}")
# def main(filepath,outpath):
    # data = np.load('/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy')
    # # Actual data shape: (118, 18, 192, 192)
    # print("Actual data shape:", data.shape)
    # data = data[100:101, :, :, :]  # 选取特定的时间帧
    # img = IFFT2c(data)  # 假设 IFFT2c 是一个定义好的函数，用于逆傅里叶变换
    # img = img[0]  # 取第一个元素，如果 IFFT2c 返回的是一个列表或数组
    # print("img:", img.shape)

    # # img_max = np.max(np.abs(img))
    # # if img_max == 0 or np.isnan(img_max):
    # #     img_norm = np.abs(img)  # 或者选择其他合理的处理方式
    # # else:
    # #     img_norm = np.abs(img) / img_max
    # img_max = np.max(np.abs(img))
    # img_norm = np.abs(img) / img_max

    # brightness_factor = 3
    # img_brightened = np.clip(img_norm * brightness_factor, 0, 1)
    # def animate(frame):
    #     plt.imshow(img_brightened[frame], cmap='gray')
    #     plt.title('Frame {}'.format(frame))
    #     plt.axis('off')

    # # 创建动画
    # anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
    # # anim.save('output-kgin_1220_r4.gif', writer='imagemagick')
    # anim.save(outpath, writer='imagemagick')
    
# data = np.load('/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy')
# /nfs/zzy/code/k_gin_base/out_kgin_vista_r8_0216_100.npy
# data = np.load('/nfs/zzy/code/k_gin_base/out_kgin_vista_r8_0216_100.npy')
# /nfs/zzy/code/k_gin_base/out_kgin_vista_r8_0216_300.npy
data = np.load('/nfs/zzy/code/k_gin_base/out_kgin_vista_r8_0216_300.npy')
    #csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[100:101,:,:,:]
data = data[0:1,:,:,:]
    #csm = csm[100,:,:,:,:] 
    #img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #

img = IFFT2c(data)
img = img[0]
print("img-shape:", img.shape)

img_max = np.max(np.abs(img))
img_norm = np.abs(img) / img_max
brightness_factor = 3
img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

def animate(frame):
    plt.imshow(img_brightened[frame], cmap='gray')  
    plt.title('Frame {}'.format(frame))
    plt.axis('off')

anim = FuncAnimation(plt.figure(), animate, frames=len(img_brightened), interval=500)
    # anim.save('output_kv01.gif', writer='imagemagick')
    # /data0/zhiyong/code/github/k-gin/out_1122.npy
    # anim.save('output_kv_kgin_1122.gif', writer='imagemagick')

    # /data0/zhiyong/code/github/itzzy_git/k-gin_kv/out_1130.npy
    # anim.save('output_kv_kgin_1130_2_1.gif', writer='imagemagick')
    # /nfs/zzy/code/k_gin_kv/output/r4/out_1220_r4.npy
    # anim.save('output_kv_kgin_1220_r4.gif', writer='imagemagick')
    # /nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy
# 100个epoch有伪影 
anim.save('out_kgin_vista_r8_0216_100.gif', writer='imagemagick')
# /nfs/zzy/code/k_gin_base/out_kgin_vista_r8_0216_300.npy  实际是vista mask
anim.save('out_kgin_vista_r8_0218.gif', writer='imagemagick')

# if __name__ == "__main__":
#     filepath = '/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy'
#     outpath = 'output-kgin_1220_r4.gif'
#     # file_path = '/nfs/zzy/code/k_gin_base/output/r4/out_1220_r4.npy'
#     # file_size = os.path.getsize(file_path)
#     # # File size: 261881856 bytes
#     # print(f"File size: {file_size} bytes")
#     main(filepath,outpath)



'''
你遇到的这些报错提示表明代码中使用了一些未正确导入的模块或函数。在Python中，如果你要使用 `plt`（`matplotlib.pyplot` 的常用缩写形式，用于绘图相关操作）以及 `FuncAnimation`（用于创建动画的函数，同样来自 `matplotlib` 库），需要先进行相应的导入操作。以下是修正后的代码示例以及相关解释：

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(frame):
    plt.imshow(img_brightened[frame], cmap='gray')  # 显示图像，这里假设img_brightened是提前定义好的图像数据列表或者数组
    plt.title('Frame {}'.format(frame))
    plt.axis('off')

# 创建动画
fig = plt.figure()  # 先创建一个Figure对象，作为动画的基础图形容器
anim = FuncAnimation(fig, animate, frames=len(img_brightened), interval=500)
# 将动画保存为GIF文件，需要确保已经安装了ImageMagick软件（如果使用'imagemagick'作为writer）
anim.save('testOF.gif', writer='imagemagick')
```

在上述代码中：

### 1. 导入相关库
- 首先通过 `import matplotlib.pyplot as plt` 导入了 `matplotlib` 库用于绘图的模块，并将其简称为 `plt`，这样后续就能使用 `plt` 来调用 `matplotlib.pyplot` 中的各种绘图函数了，比如 `imshow`（用于显示图像）、`title`（设置图像标题）、`axis`（控制坐标轴显示等）。
- 然后通过 `from matplotlib.animation import FuncAnimation` 单独导入了用于创建动画的 `FuncAnimation` 函数，它需要传入一个 `Figure` 对象（通过 `plt.figure()` 创建）、一个更新每一帧画面的函数（这里是 `animate` 函数）以及其他相关参数（如动画的帧数、帧与帧之间的时间间隔等）来生成动画对象。

### 2. `animate` 函数
这个函数定义了每一帧画面要显示的内容，在这里它接收一个表示帧序号的参数 `frame`，然后使用 `plt.imshow` 展示 `img_brightened` 中对应帧的图像数据（以灰度模式显示，通过 `cmap='gray'` 指定），接着设置图像标题展示当前是第几帧，最后关闭坐标轴显示（通过 `axis('off')`），使得图像看起来更简洁，专注于展示图像本身内容。

### 3. 创建和保存动画
- 先使用 `plt.figure()` 创建了一个空白的 `Figure` 对象 `fig`，这个对象作为整个动画的载体，后续的动画元素都会添加到这个图形上。
- 接着使用 `FuncAnimation` 函数创建动画对象 `anim`，传入前面创建的 `fig`、定义好的 `animate` 函数以及指定动画的帧数（通过 `len(img_brightened)` 获取图像数据的帧数）和帧间隔时间（`interval=500` 表示每帧间隔500毫秒）等参数。
- 最后通过 `anim.save` 方法将创建好的动画保存为名为 `testOF.gif` 的GIF文件，这里指定了使用 `'imagemagick'` 作为写入器（需要提前安装好 `ImageMagick` 软件才能正常使用这个写入器来生成GIF动画，如果没安装可以考虑使用其他支持的写入器，比如 `'pillow'` 等，使用方式类似）。

请确保在运行代码之前，已经正确安装了 `matplotlib` 库及其相关依赖（例如在命令行中使用 `pip install matplotlib` 命令安装，具体取决于你使用的Python环境管理工具，如 `conda` 环境则使用对应的 `conda` 安装命令），并且如果使用 `'imagemagick'` 作为写入器，要安装好 `ImageMagick` 软件。另外，还要保证代码中的 `img_brightened` 变量已经在前面的代码中正确定义并赋值为有效的图像数据（比如是一个形状合适的多维数组，其第一维表示帧数等符合动画要求的结构）。

希望这些解释和示例能帮助你解决代码中的报错问题，顺利实现图像动画的创建和保存功能。 
'''