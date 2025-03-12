import scipy.io as sio
import numpy as np
import os
import h5py
import torch
from pyexcel_ods import get_data
from termcolor import colored
from scipy.io import savemat
from scipy.ndimage.morphology import distance_transform_edt
from utils.mri_related import fft2c, MulticoilAdjointOp

'''
这段代码处理多通道k空间数据，进行图像重建，并生成各种掩码用于图像处理或模型训练。 
它使用了多种库，包括 scipy.io, h5py, numpy, torch 和
一些自定义函数 (例如 MulticoilAdjointOp, fft2c)。
代码中的一些函数 (例如 get_data) 没有给出定义，需要根据上下文补充。 
tracemalloc 可以用来分析这段代码中 ToTorchIO 函数的内存使用情况，特别是当处理大规模数据时。
注意观察 torch.from_numpy 的内存占用。
'''

'''
img = MulticoilAdjointOp(...): 这是核心重建步骤，将多通道k空间数据和线圈灵敏度图 (coilmaps) 转换为单通道图像。
center=True 可能表示中心化处理，coil_axis=-4 指定线圈维度，channel_dim_defined=False 可能表示通道维度未明确定义。
img /= img.abs().max(): 将图像的像素值归一化到最大绝对值为1。
kspace = fft2c(img): 将图像转换回k空间。fft2c 很可能是二维快速傅里叶变换的自定义函数。

这行代码的核心是调用了一个名为 MulticoilAdjointOp 的函数或类，它执行多通道磁共振图像(MRI)重建。让我们逐部分解读：

- MulticoilAdjointOp(center=True, coil_axis=-4, channel_dim_defined=False)**:**  这部分创建了一个 MulticoilAdjointOp 对象，并设置了三个参数：
    * center=True:  这表示重建过程中会进行中心化处理。在 MRI 数据处理中，中心化通常指将k空间数据或图像数据移到其中心位置，
    这有助于减少重建伪影。

    * coil_axis=-4:  这指定了线圈维度的索引。  -4 表示线圈维度是张量的倒数第四个维度。 
    这取决于输入张量的形状，例如，如果 kspace 的形状是 (batch_size, channels, height, width, complex)，
    那么 coil_axis=-4 就表示 channels 维度。  不同的 MRI 数据格式可能会有不同的维度顺序，因此这个参数非常重要。

    * channel_dim_defined=False:  这表示输入数据中线圈维度是否已经明确定义。如果设置为 True，
    则 MulticoilAdjointOp  会假设输入数据已经按照特定的方式组织了线圈维度；
    如果设置为 False，则 MulticoilAdjointOp  需要根据其他信息（例如，coilmaps）来确定线圈维度。


- (kspace, torch.ones_like(kspace), coilmaps)**:**  这是对 MulticoilAdjointOp 对象的调用，它接受三个参数作为输入：

    * kspace:  这是多通道 k 空间数据，它是重建的输入。  k 空间数据是 MRI 扫描的原始数据，它包含了图像的频率信息。

    * torch.ones_like(kspace):  这创建一个与 kspace 形状相同的全1张量。 
    这个张量很可能用作权重或掩码，在重建过程中对不同位置的 k 空间数据赋予不同的权重。
    具体作用取决于 MulticoilAdjointOp 的内部实现。

    * coilmaps:  这是线圈灵敏度图。线圈灵敏度图描述了每个线圈在图像不同位置的灵敏度，
    它用于补偿不同线圈对图像信号的贡献差异，从而提高重建图像的质量。


- img = ...**:**  最终，MulticoilAdjointOp 的输出赋值给变量 img。 
img  是一个单通道的重建图像，它代表了从多通道 k 空间数据中重建出来的图像。
总结:
这行代码使用 MulticoilAdjointOp 函数或类对多通道 k 空间数据进行重建，得到单通道图像。  
它利用线圈灵敏度图 (coilmaps) 来补偿不同线圈的灵敏度差异，并使用中心化处理来减少重建伪影。  
torch.ones_like(kspace) 的作用需要根据 MulticoilAdjointOp 的具体实现来确定，它可能用于加权或掩码操作。 
这行代码是多通道 MRI 重建中的一个关键步骤。  
MulticoilAdjointOp 很可能是一个基于某种算法（例如，最小二乘法或压缩感知）实现的自定义函数或类。
'''
# def multicoil2single(kspace, coilmaps):
#     img = MulticoilAdjointOp(center=True, coil_axis=-4, channel_dim_defined=False)(kspace, torch.ones_like(kspace), coilmaps)
#     img /= img.abs().max()
#     kspace = fft2c(img)
#     return kspace, img

def multicoil2single(kspace, coilmaps):
    img = MulticoilAdjointOp(center=True, coil_axis=-4, channel_dim_defined=False)(kspace, torch.ones_like(kspace), coilmaps)
    # img = MulticoilAdjointOp(center=False, coil_axis=-4, channel_dim_defined=False)(kspace, torch.ones_like(kspace), coilmaps)
    img /= img.abs().max()
    kspace = fft2c(img)
    return kspace, img


class ToTorchIO():
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, sample):
        inputs = []
        outputs = []
        for key in self.input_keys:
            inputs.append(torch.from_numpy(sample[key]))
        for key in self.output_keys:
            outputs.append(torch.from_numpy(sample[key]))
        return inputs, outputs


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


def get_valid_slices(valid_slice_info_file, data_type='img', central_slice_only=True):
    valid_slice_info = get_data(valid_slice_info_file)
    ods_col = 2 if data_type == 'img' else 1  # for 'img', ods_col = 2, while for 'dicom', ods_col = 1
    if central_slice_only: ods_col = 3
    valid_slices = {value[0]: list(
        range(*[int(j) - 1 if i == 0 else int(j) for i, j in enumerate(value[ods_col].split(','))])) for value in
        valid_slice_info["Sheet1"][1:] if len(value) != 0}
    return valid_slices


def load_mask(mask_root, nPE, acc_rate, pattern='VISTA'):
    if acc_rate != 1:
        mask_path = os.path.join(mask_root, pattern, f"mask_VISTA_{nPE}x25_acc{acc_rate}_demo.txt")
        mask = np.loadtxt(mask_path, dtype=np.int32, delimiter=",")
    else:
        mask = np.ones((nPE, 25), dtype=np.int32)
    mask = np.transpose(mask)[None, :, None, :]
    return mask


def get_bounding_box_value(path, image_size, offset=10):
    """
    the 8 dims of the box_info are: [xmax, xmin, xmean, xstd, ymax, ymin, ymin, ystd]
    :param path:
    :param image_size:
    :param offset:
    :return:
    """
    box_info = np.load(path)['box_info']
    slc_num = box_info.shape[0]
    xmax = [min(box_info[slc, 0] + offset, image_size[0]) for slc in range(slc_num)]
    xmin = [max(box_info[slc, 1] - offset, 0) for slc in range(slc_num)]
    ymax = [min(box_info[slc, 4] + offset, image_size[1]) for slc in range(slc_num)]
    ymin = [max(box_info[slc, 5] - offset, 0) for slc in range(slc_num)]
    return np.array([xmin, xmax, ymin, ymax])


def generate_weighting_mask(image_size, boundary, mode):
    [xmin, xmax, ymin, ymax] = boundary
    if mode == 'hard':
        box = np.zeros(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 1
    elif mode == 'exp_decay':
        box = np.ones(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 0
        box = distance_transform_edt(box) / 4  # laplace weighting decay
        box = np.exp(-0.2*box).astype(np.float32)

    elif mode == 'all':
        box = np.ones(image_size, dtype=np.uint8)
    else:
        raise TypeError('wrong mode type is given')
    return box


def get_bounding_box(box_path, image_size, slc, offset=10, mode='hard'):
    """
    older version: combination of get_bounding_box_value and generate_weighting_mask. And this function can only deal with one single slice.
    :param path:
    :param z:
    :param offset:
    :param mode: mode can be 'hard': within the box = 1, otherwise 0. Or 'exp': within the box = 1, otherwise
                exponential decay beginning at the boundary
    :return:
    """
    # box_info = np.load(path)['box_info']
    box_info = np.load(box_path)['box_info']
    xmax, xmin = min(box_info[slc, 0] + offset, image_size[0]), max(box_info[slc, 1] - offset, 0)
    ymax, ymin = min(box_info[slc, 4] + offset, image_size[1]), max(box_info[slc, 5] - offset, 0)
    if mode == 'hard':
        box = np.zeros(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 1
    elif mode == 'exp_decay':
        box = np.ones(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 0
        box = distance_transform_edt(box) / 4  # laplace weighting decay
        box = np.exp(-0.2*box).astype(np.float32)

    elif mode == 'all':
        box = np.ones(image_size, dtype=np.uint8)
    else:
        raise TypeError('wrong mode type is given')
    return box, [xmin, xmax, ymin, ymax]


def h52mat(load_path, save_path):
    with h5py.File(load_path, 'r') as ds:
        image = np.array(ds['dImgC'])
        image = np.squeeze(image)
        image = np.transpose(image, (3,2,0,1))
        savemat(save_path, {'dImgC': image})
