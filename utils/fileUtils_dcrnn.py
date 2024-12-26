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
data = np.load('/nfs/zzy/code/k_gin_base/output/r6/out_1206_1.npy')
# data = np.load('/nfs/zzy/code/k_gin_base/models/dc_rnn_test3/im_recon.npy')



#csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')
# csm = np.load('/data0/chentao/data/LplusSNet/data/20coil/csm_cine_multicoil_test.npy')

# data: (118, 18, 192, 192)
# img: (18, 192, 192)
# data: (118, 2, 192, 192, 18)
print("data:", data.shape) #data: (800, coil=20, 18, 192, 192) (t,h,w)=(18, 192, 192)
# data = data[100:101,:,:,:]
data = data[100:101,:,:,:]
#csm = csm[100,:,:,:,:]
# csm = csm[100:101,:,:,:] 
# print('csm:',csm.shape)

#img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #
# img = np.sum(IFFT2c(data) * np.conj(csm), axis=0) #

img = IFFT2c(data)
print('img:',img.shape)
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
# dc_rnn_test3
# anim.save('output-dc_rnn_test3.gif', writer='imagemagick')


# # 加载数据
# data = np.load('/nfs/zzy/code/k_gin_base/models/dc_rnn_test3/im_recon.npy')

# # 打印数据形状
# print("data:", data.shape)  # data: (118, 2, 192, 192, 18)

# # 选择特定的 batch
# data = data[100:101, :, :, :, :]  # 形状变为 (1, 2, 192, 192, 18)

# # 调整 data 的形状
# data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])  # 从 (1, 2, 192, 192, 18) 变为 (2, 192, 192, 18)

# # 调用 IFFT2c 函数
# img = IFFT2c(data)  # 输出形状为 (2, 192, 192)
# print('img:',img.shape)
# # 恢复 img 的形状
# img = img.reshape(1, 2, 192, 192)  # 从 (2, 192, 192) 变为 (1, 2, 192, 192)

# # 归一化图像数据
# img_max = np.max(np.abs(img))
# if img_max == 0 or np.isnan(img_max):
#     img_norm = np.abs(img)  # 如果 img_max 为零或 NaN，直接取绝对值
# else:
#     img_norm = np.abs(img) / img_max  # 归一化

# # 调整亮度
# brightness_factor = 3
# img_brightened = np.clip(img_norm * brightness_factor, 0, 1)

# # 定义动画函数
# def animate(frame):
#     plt.imshow(img_brightened[0, frame], cmap='gray')  # 显示第 0 个 batch 的第 frame 帧
#     plt.title('Frame {}'.format(frame))
#     plt.axis('off')

# # 创建动画
# anim = FuncAnimation(plt.figure(), animate, frames=img_brightened.shape[1], interval=500)

# # 保存动画
# anim.save('output-dc_rnn_test3.gif', writer='imagemagick')






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