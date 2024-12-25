"""
版权声明：开头的注释部分表明了代码的版权归属以及遵循的 MIT 许可协议，告知使用者代码的授权情况。
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

'''
模块导入：
math：Python 的标准数学库，在这里可能用于一些数值计算，例如在后续处理图像尺寸的补齐计算中会用到取整等数学运算。
从 typing 模块导入了 List、Optional、Tuple 类型提示相关的类，用于在 Python 代码中更清晰地指定函数参数和返回值的类型，
增强代码的可读性和可维护性，方便开发者以及代码阅读者理解变量的数据结构预期。
torch、torch.nn（以 nn 别名导入）、torch.nn.functional（以 F 别名导入）：PyTorch 的核心模块，
用于构建神经网络、定义网络层以及实现常见的神经网络相关的函数操作（如各种激活函数、卷积、池化等操作的函数形式）。
fastmri：应该是一个与 MRI 相关的自定义或外部库，可能包含了针对 MRI 数据处理、变换以及特定指标计算等功能的函数和类，
例如后续代码中会用到其中的 rss_complex、ifft2c 等函数来处理 MRI 图像数据在频域和空域之间的转换以及计算均方根等操作。
from fastmri.data import transforms：从 fastmri 库的数据模块中导入 transforms，
这个模块可能包含了对 MRI 数据进行各种变换（如掩码操作、归一化等）的函数，
在代码中用于处理 k - 空间数据等情况，以符合模型输入的要求。
from.unet import Unet：从当前目录下的 unet 模块（文件）中导入 Unet 类，Unet 是一种常见的神经网络架构，
在这里作为构建其他模型的基础组件，用于特征提取、图像重建等任务。
'''
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
# from fastmri.data import transforms

from unet import Unet

class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    对输入进行归一化处理后再通过U-Net的模型结构
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NormUnet------ torch.Size([2, 18, 192, 192])
        print('NormUnet------',x.shape)
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    简化后的模型，由于输入是单线圈数据，不再需要复杂的多线圈灵敏度估计相关逻辑
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, num_low_frequencies: Optional[int] = None):
        # 直接对输入的masked_kspace进行处理，不再涉及多线圈转换等操作
        x = self.norm_unet(masked_kspace)
        return x


class VarNet(nn.Module):
    """
    适配单线圈输入的修改后的VarNet模型
    """

    # def __init__(self, num_cascades: int = 12, chans: int = 18, pools: int = 4):
    #     """
    #     修改后的初始化函数，去除了与多线圈灵敏度图相关的参数，只保留核心的级联层数、U-Net通道数和池化层数等参数
    #     """
    #     super().__init__()
    #     self.cascades = nn.ModuleList(
    #         [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
    #     )
    def __init__(self, config):
        super().__init__()
        config = config.VarNet
        self.num_cascades = config.num_cascades
        self.chans = config.chans
        self.pools = config.pools
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(self.chans, self.pools)) for _ in range(self.num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        print('vatNet forward-------')
        # vatNet forward-masked_kspace torch.Size([2, 18, 192, 192])
        print('vatNet forward-masked_kspace', masked_kspace.shape)
        masked_kspace_orig = torch.view_as_real(masked_kspace)
        print('vatNet forward-masked_kspace_orig', masked_kspace_orig.shape)
        mask_orig = mask[..., None,None].expand_as(masked_kspace_orig)
        # mask_orig = mask[..., None].expand_as(masked_kspace)  # 将掩码扩展到与图像相同的维度
        # vatNet forward-------mask_orig torch.Size([2, 18, 192, 192, 2])
        # vatNet forward-------mask_orig torch.Size([2, 18, 192, 192])
        print('vatNet forward-------mask_orig',mask_orig.shape) 
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            # 直接传入当前的kspace_pred、masked_kspace和mask进行计算，不再需要sens_maps参数
            # kspace_pred = cascade(kspace_pred, masked_kspace, mask)
            kspace_pred = cascade(kspace_pred, masked_kspace, mask_orig)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


class VarNetBlock(nn.Module):
    """
    适配单线圈输入的模型块，去除了与sens_maps相关操作
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # cascade(kspace_pred, masked_kspace, mask_orig)
        # VarNetBlock forward-------current_kspace torch.Size([2, 18, 192, 192])
        # VarNetBlock forward-------ref_kspace torch.Size([2, 18, 192, 192])
        # VarNetBlock forward-------mask torch.Size([2, 18, 192, 192, 2])
        print('VarNetBlock forward-------current_kspace',current_kspace.shape)
        print('VarNetBlock forward-------ref_kspace',ref_kspace.shape)
        print('VarNetBlock forward-------mask',mask.shape)        
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        mask = mask[..., 0]  # 移除最后一个维度
        # zero = torch.zeros(current_kspace.shape, device=current_kspace.device)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.model(current_kspace)  # 这里不再需要基于sens_maps的相关操作来获取model_term

        return current_kspace - soft_dc - model_term

'''
NormUnet类定义了一个带有归一化操作的 U-Net 模型结构。它在常规 U-Net 的基础上，
在输入进入U-Net 之前对数据进行归一化处理，目的是在训练过程中使数据值在数值上更加稳定，
避免出现梯度消失或爆炸等数值不稳定问题，同时提供了一些辅助的数据维度变换、填充和反填充等操作方法，
以适应 MRI 数据的处理和 U-Net 的输入输出要求。
'''
# class NormUnet(nn.Module):
#     """
#     Normalized U-Net model.

#     This is the same as a regular U-Net, but with normalization applied to the
#     input before the U-Net. This keeps the values more numerically stable
#     during training.
#     """
    
#     '''
#     类的初始化方法 __init__ 解读:
#     在初始化方法中，首先调用父类（nn.Module）的初始化方法 super().__init__()，确保父类的初始化逻辑得以执行
#     （例如注册模型的参数等操作）。然后创建了一个 Unet 实例并赋值给 self.unet，传入了多个参数来配置 Unet 的结构，
#     包括输入通道数 in_chans（默认值为 2，可能对应 MRI 数据的某种双通道表示形式，比如实部和虚部等情况）、
#     输出通道数 out_chans（同样默认值为 2，根据具体应用场景确定输出的数据通道结构）、
#     第一个卷积层的输出通道数 chans（控制网络的特征提取通道数规模，影响模型的表达能力）、
#     下采样和上采样的层数 num_pool_layers（决定网络的深度和特征图尺寸变化情况）以及 drop_prob（Dropout 概率，
#     用于在训练时随机丢弃部分神经元连接，防止过拟合，这里默认值为 0 表示不使用 Dropout）。
#     '''
#     def __init__(
#         self,
#         chans: int,
#         num_pools: int,
#         in_chans: int = 2,
#         out_chans: int = 2,
#         drop_prob: float = 0.0,
#     ):
#         """
#         Args:
#             chans: Number of output channels of the first convolution layer.
#             num_pools: Number of down-sampling and up-sampling layers.
#             in_chans: Number of channels in the input to the U-Net model.
#             out_chans: Number of channels in the output to the U-Net model.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()

#         self.unet = Unet(
#             in_chans=in_chans,
#             out_chans=out_chans,
#             chans=chans,
#             num_pool_layers=num_pools,
#             drop_prob=drop_prob,
#         )
#     '''
#     数据维度变换相关方法解读:
#     这个方法用于将输入张量 x 的维度进行变换，将原本表示复数的最后一个维度（维度大小应该为 2，对应复数的实部和虚部）
#     变换到通道维度之前，方便后续的归一化以及 U-Net 处理。首先通过 assert 语句断言最后一个维度大小为 2，
#     确保输入数据符合预期的复数表示形式。然后使用 permute 函数对维度顺序进行重新排列，
#     将最后一个维度（索引为 4）移动到索引为 1 的位置，接着使用reshape函数将新的维度结构调整为将复数的实部和虚部合并到通道维度
#     （通道数变为原来的 2 倍），最终返回维度变换后的张量。
#     '''
#     def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, h, w, two = x.shape
#         assert two == 2
#         return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

#     '''
#     数据维度变换:
#     该方法与 complex_to_chan_dim 方法相反，是将经过处理后的张量 x 的维度再转换回将复数表示放在最后一个维度的形式，
#     以便输出符合特定要求的数据结构。首先通过 assert 断言当前通道维度的大小 c2 是偶数，
#     因为它应该是之前合并实部和虚部后的结果，需要能平均拆分为实部和虚部两部分。然后计算出原始的通道数 c，
#     接着使用 view 函数将张量的维度调整为将实部和虚部拆开的形式（增加一个维度表示实部和虚部），
#     再通过 permute 函数将复数维度（实部和虚部所在维度）移动到最后一个位置，
#     最后使用 contiguous 函数确保张量的内存布局是连续的（有些情况下经过维度变换后内存可能不连续，
#     这会影响后续的一些操作，调用此函数可避免潜在问题），并返回维度转换后的张量。
#     '''
#     def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
#         b, c2, h, w = x.shape
#         assert c2 % 2 == 0
#         c = c2 // 2
#         return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
#     '''
#     归一化及反归一化相关方法解读:
#     此方法实现了一种分组归一化（Group Norm）操作。首先获取输入张量 x 的形状信息（批量大小 b、通道数 c、高度 h、宽度 w），
#     然后将其维度调整为 (b, 2, c // 2 * h * w) 的形式，这里将通道维度拆分为两部分（可能是考虑到复数数据的实部和虚部情况，
#     或者是按照某种分组规则进行归一化的准备），接着分别沿着新的维度（索引为 2 的维度）计算均值 mean 和标准差 std，
#     并将它们的形状调整为 (b, 2, 1, 1)，方便后续进行归一化操作时可以对每个样本、每组数据（这里按实部和虚部等分组概念）
#     进行广播式的归一化计算。最后再将 x 的维度恢复为原来的 (b, c, h, w) 形式，并返回归一化后的 x 
#     以及计算得到的均值 mean 和标准差 std，这样在后续需要反归一化时可以使用这些保存的值进行还原操作。
#     '''
#     def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # group norm
#         b, c, h, w = x.shape
#         x = x.view(b, 2, c // 2 * h * w)

#         mean = x.mean(dim=2).view(b, 2, 1, 1)
#         std = x.std(dim=2).view(b, 2, 1, 1)

#         x = x.view(b, c, h, w)

#         return (x - mean) / std, mean, std
#     '''
#     这是与 norm 方法对应的反归一化方法，根据传入的归一化后的张量 x 以及之前保存的均值 mean 和标准差 std，
#     通过简单的线性变换（乘以标准差再加上均值）将数据还原到归一化之前的状态，用于在模型的后续处理步骤中，
#     当需要将经过归一化处理的数据恢复到原始的数据尺度时进行操作。
#     '''
#     def unnorm(
#         self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
#     ) -> torch.Tensor:
#         return x * std + mean
#     '''
#     数据填充及反填充相关方法解读:
#     该方法用于对输入张量 x 进行填充操作，目的可能是为了使图像数据的尺寸满足后续 U - Net 等操作对尺寸的要求
#     （例如要求尺寸是某个特定值的倍数等情况）。首先获取输入张量的高度 h 和宽度 w，
#     然后通过一些位运算和计算（((w - 1) | 15) + 1 和 ((h - 1) | 15) + 1，
#     这里的计算逻辑是将宽度和高度向上调整到最接近的 16 的倍数，具体是利用了位运算的特性来快速计算，
#     目的是为了满足一些硬件加速或者网络结构对输入尺寸的优化要求）得到填充后的目标宽度 w_mult 和高度 h_mult。
#     接着计算在宽度和高度方向上需要填充的左右和上下的像素数量（通过取整操作分别得到左右和上下的填充量），
#     并将这些填充量组合成一个列表 w_pad + h_pad，最后使用 torch.nn.functional.pad 函数
#     （这里文档说明和实际使用存在一些类型不一致的情况，代码中注释提到了相关问题，实际传入的是列表形式的填充参数）
#     对张量 x 按照计算得到的填充量进行填充操作，并返回填充后的张量 x 以及包含填充信息的元组
#     （h_pad、w_pad、h_mult、w_mult），这些填充信息在后续的反填充操作中会用到。
#     '''
#     def pad(
#         self, x: torch.Tensor
#     ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
#         _, _, h, w = x.shape
#         w_mult = ((w - 1) | 15) + 1
#         h_mult = ((h - 1) | 15) + 1
#         w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
#         h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
#         # TODO: fix this type when PyTorch fixes theirs
#         # the documentation lies - this actually takes a list
#         # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
#         # https://github.com/pytorch/pytorch/pull/16949
#         x = F.pad(x, w_pad + h_pad)

#         return x, (h_pad, w_pad, h_mult, w_mult)
#     '''
#     方法功能概述：
#     这个方法的作用是对之前经过填充（pad 方法操作）的张量进行反填充操作，也就是去除之前添加的填充像素，
#     将张量的尺寸恢复到接近原始输入的大小（除去为了满足特定计算要求而添加的填充部分），
#     使得数据能够以合适的尺寸继续后续的处理流程或者作为最终的输出。
#     '''
#     '''
#     参数含义及操作逻辑解读：
#     x：是需要进行反填充操作的输入张量，它应该是之前已经经过了填充操作的张量，
#     其维度结构通常符合经过 pad 方法处理后的形式，并且包含了额外的填充像素信息。
#     h_pad：是一个包含两个整数的列表，表示在高度方向上的填充信息，其中 h_pad[0] 是高度方向上前面（顶部）填充的像素数量，
#     h_pad[1] 是高度方向上后面（底部）填充的像素数量，这个信息是在之前调用 pad 方法进行填充操作时记录下来的，
#     用于准确知道要去除哪些填充像素。
#     w_pad：同样是一个包含两个整数的列表，对应宽度方向上的填充信息，w_pad[0] 表示宽度方向上左边填充的像素数量，
#     w_pad[1] 表示宽度方向上右边填充的像素数量，用于确定在宽度方向上需要去除的填充部分。
#     h_mult 和 w_mult：分别表示填充后高度和宽度方向上的目标尺寸（通常是经过向上取整到特定倍数后的尺寸，
#     例如之前 pad 方法中计算得到的最接近 16 的倍数等情况），在计算去除填充后的尺寸范围时会用到这些值。
#     操作逻辑上，通过使用 Python 的切片语法 [..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]] 
#     对输入张量 x 进行操作。... 表示对前面的所有维度进行通配（保持不变），然后在高度维度上，从 h_pad[0] 这个位置开始，
#     取到 h_mult - h_pad[1] 这个位置之前（即去除了顶部和底部的填充部分），在宽度维度上，从 w_pad[0] 位置开始，
#     取到 w_mult - w_pad[1] 位置之前（去除了左边和右边的填充部分），最终返回经过反填充操作后的张量，
#     其尺寸已经恢复到去除填充像素后的合适大小，以便后续继续处理或者输出符合要求的数据形式。
#     '''
#     def unpad(
#         self,
#         x: torch.Tensor,
#         h_pad: List[int],
#         w_pad: List[int],
#         h_mult: int,
#         w_mult: int,
#     ) -> torch.Tensor:
#         return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
#     '''
#     方法功能概述：
#     forward方法是NormUnet 类的前向传播方法，定义了数据在这个带有归一化的U-Net模型中的完整正向流动过程，
#     包括了数据维度的调整、归一化、填充、通过Unet进行特征提取、反填充、反归一化以及最终的数据维度恢复等一系列操作，
#     使得输入数据经过这一系列处理后能够得到符合要求的输出结果，整个过程整合了之前定义的各个辅助方法来实现规范化的模型前向计算逻辑。
#     '''
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         '''
#         输入数据维度检查：
#         首先对输入张量x的最后一个维度进行检查，通过判断 x.shape[-1] 是否等于2来确保输入数据的最后一维大小为 2，
#         因为在这个模型的设计中，最后一维是用于表示复数（通常复数用实部和虚部两个值表示，所以维度大小为 2），
#         如果不符合这个要求，则抛出 ValueError 异常，提示用户输入的数据格式不符合模型期望的复数表示形式，
#         这样可以提前发现数据格式错误，避免后续在基于复数相关操作的处理中出现问题。
#         '''
#         if not x.shape[-1] == 2:
#             raise ValueError("Last dimension must be 2 for complex.")

#         # get shapes for unet and normalize 准备 U - Net 输入并归一化数据：
#         '''
#         x = self.complex_to_chan_dim(x)：调用 complex_to_chan_dim 方法将输入张量 x 的维度进行变换，
#         把原本表示复数的最后一个维度（维度大小为 2）变换到通道维度之前，方便后续的归一化以及 Unet 处理，
#         具体的维度变换逻辑在 complex_to_chan_dim 方法中已经介绍过，经过这一步操作后，数据的维度结构更符合后续处理的要求。
#         x, mean, std = self.norm(x)：接着调用 norm 方法对维度调整后的 x 进行归一化操作，这里采用的是分组归一化
#         （Group Norm）的方式，计算并返回归一化后的 x 以及对应的均值 mean 和标准差 std，这些值在后续的反归一化操作中会用到，
#         归一化的目的是使数据在数值上更加稳定，有利于模型的训练和避免梯度相关的数值问题。
#         x, pad_sizes = self.pad(x)：再调用 pad 方法对归一化后的 x 进行填充操作，
#         使图像数据的尺寸满足后续 Unet 等操作对尺寸的要求（例如尺寸是某个特定值的倍数等情况），
#         同时返回填充后的 x 以及包含填充信息的 pad_sizes（包含高度和宽度方向的填充量以及填充后的目标尺寸等信息），
#         填充后的 x 将作为 Unet 的输入。
#         '''
#         x = self.complex_to_chan_dim(x)
#         x, mean, std = self.norm(x)
#         x, pad_sizes = self.pad(x)
#         '''
#         通过 U - Net 进行特征提取：
#         将经过前面维度调整、归一化和填充后的张量x输入到self.unet（也就是在 __init__ 方法中初始化的 Unet 实例）中，
#         利用Unet的网络结构进行特征提取和处理，Unet会根据其内部定义的卷积、池化等操作对输入数据进行层层处理，
#         挖掘数据中的特征信息，最终输出经过特征提取后的结果，这个结果仍然保持着经过填充后的尺寸等信息，
#         后续还需要进行一些恢复操作来得到合适的最终输出。
#         x = self.unpad(x, *pad_sizes)：调用 unpad 方法，传入填充信息 pad_sizes 对经过 Unet 处理后的张量 x 进行反填充操作，
#         去除之前添加的填充像素，将数据的尺寸恢复到接近原始输入的大小（除去为了满足特定计算要求而添加的填充部分），
#         使得数据能够以合适的尺寸继续后续的处理流程或者作为最终的输出。
#         x = self.unnorm(x, mean, std)：接着调用unnorm方法，根据之前保存的均值mean和标准差std对反填充后的x
#         进行反归一化操作，通过线性变换（乘以标准差再加上均值）将数据还原到归一化之前的状态，
#         以得到符合实际数据尺度和后续使用要求的结果。
#         x = self.chan_complex_to_last_dim(x)：最后调用chan_complex_to_last_dim方法
#         将张量x的维度再转换回将复数表示放在最后一个维度的形式，恢复到符合特定要求的数据结构，
#         以便输出符合模型整体设计和后续处理期望的数据形式。
#         '''
#         x = self.unet(x)

#         # get shapes back and unnormalize
#         x = self.unpad(x, *pad_sizes)
#         x = self.unnorm(x, mean, std)
#         x = self.chan_complex_to_last_dim(x)
#         '''
#         经过前面一系列的操作后，将处理好的张量 x 作为 forward 方法的结果返回，这个返回的张量就是NormUnet
#         模型对输入数据进行前向传播处理后的最终输出，它可以作为后续模型（例如整个 VarNet 模型中其他模块）的输入，
#         或者用于计算损失、与真实标签对比等进一步的操作中，具体取决于模型所处的整体架构和应用场景。
#         总的来说，forward 方法按照特定的顺序和逻辑调用了 NormUnet 类中定义的多个辅助方法，
#         实现了对输入数据从预处理、特征提取到后处理恢复形状等完整的正向计算过程，
#         确保数据在模型中能够正确地流动并得到符合要求的输出结果，是整个NormUnet模型在实际运行中进行推理或训练时的核心数据处理流程。
#         '''
#         return x

# '''
# SensitivityModel类用于从k-空间数据中学习敏感度估计。它先将多通道的k-空间数据进行逆快速傅里叶变换（IFFT）转换到图像空间，
# 然后利用 NormUnet 对线圈图像进行处理来估计线圈敏感度，并且提供了一些数据维度变换以及与掩码处理、
# 计算低频线数量和填充等相关的辅助方法，在端到端变分网络中起着重要作用。
# '''
# class SensitivityModel(nn.Module):
#     """
#     Model for learning sensitivity estimation from k-space data.

#     This model applies an IFFT to multichannel k-space data and then a U-Net
#     to the coil images to estimate coil sensitivities. It can be used with the
#     end-to-end variational network.
#     """
#     '''
#     在初始化方法中，首先调用父类（nn.Module）的初始化方法 super().__init__()。
#     接着初始化一个布尔类型的属性 self.mask_center，用于指示是否在敏感度图计算时对k-空间的中心进行掩码处理。
#     然后创建一个 NormUnet 实例并赋值给 self.norm_unet，传入相应的参数来配置 NormUnet 的结构
#     （如通道数、池化层数等，作用与之前介绍 NormUnet 类初始化时类似），
#     这个NormUnet将用于后续对转换到图像空间后的线圈图像进行处理，以估计敏感度。
#     '''
#     def __init__(
#         self,
#         chans: int,
#         num_pools: int,
#         in_chans: int = 2,
#         out_chans: int = 2,
#         drop_prob: float = 0.0,
#         mask_center: bool = True,
#     ):
#         """
#         Args:
#             chans: Number of output channels of the first convolution layer.
#             num_pools: Number of down-sampling and up-sampling layers.
#             in_chans: Number of channels in the input to the U-Net model.
#             out_chans: Number of channels in the output to the U-Net model.
#             drop_prob: Dropout probability.
#             mask_center: Whether to mask center of k-space for sensitivity map
#                 calculation.
#         """
#         super().__init__()
#         self.mask_center = mask_center
#         self.norm_unet = NormUnet(
#             chans,
#             num_pools,
#             in_chans=in_chans,
#             out_chans=out_chans,
#             drop_prob=drop_prob,
#         )
#     '''
#     数据维度变换相关方法解读：
#     该方法用于将输入张量 x 的通道维度合并到批量维度，改变数据的维度结构，方便后续操作
#     （例如在某些情况下可以将多通道数据当作多个独立的样本一起处理）。
#     首先获取输入张量 x 的形状信息（批量大小 b、通道数 c、高度 h、宽度 w、复数维度 comp，
#     这里 comp 通常为 2 表示复数的实部和虚部），然后通过 view 函数将其维度调整为 (b * c, 1, h, w, comp) 的形式，
#     即将通道维度的大小 c 与批量大小 b 相乘，把通道维度合并到批量维度，同时新增一个通道维度大小为 1 的维度，
#     最后返回维度变换后的张量以及原始的批量大小 b，原始批量大小 b 的值后续在恢复维度结构时可能会用到。
#     '''
#     def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
#         b, c, h, w, comp = x.shape

#         return x.view(b * c, 1, h, w, comp), b
#     '''
#     此方法与 chans_to_batch_dim 相反，是将之前合并到批量维度的通道维度再恢复回来。首先获取输入张量 x 的形状信息
#     （经过变换后新的批量大小与通道数合并后的维度 bc、高度 h、宽度 w、复数维度 comp），
#     然后通过计算 bc // batch_size 得到原始的通道数 c（这里 batch_size 是之前记录的原始批量大小，用于还原正确的维度结构），
#     最后使用 view 函数将张量的维度调整为 (batch_size, c, h, w, comp) 的形式，
#     恢复到将批量维度和通道维度分开的原始数据维度结构，并返回维度转换后的张量。
#     '''
#     def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
#         bc, _, h, w, comp = x.shape
#         c = bc // batch_size

#         return x.view(batch_size, c, h, w, comp)
#     '''
#     计算相关辅助方法解读:
#     这个方法用于将输入张量x除以其在指定维度（这里是维度 1）上的均方根（Root Sum of Squares，RSS）值，
#     实现一种归一化或者调整数据幅度的操作。首先调用 fastmri.rss_complex 函数
#     （假设这个函数是 fastmri 库中用于计算复数数据的均方根的函数）计算 x 在维度 1 上的均方根值，
#     然后通过 unsqueeze 函数分别在最后一个维度和倒数第二个维度上增加维度
#     （将原本的标量值扩展为可以与 x 进行广播除法运算的张量形状），最后将 x 除以这个扩展后的均方根张量，
#     返回计算后的结果，这样可以使得数据在某种程度上进行归一化处理，便于后续作为敏感度估计等操作的结果。
#     '''
#     def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
#         return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

#     '''
#     此方法用于根据给定的掩码 mask 和可选的低频线数量参数 num_low_frequencies 来计算填充量pad以及实际的低频线数量
#     num_low_frequencies_tensor。如果 num_low_frequencies 为 None 或者等于 0，
#     意味着需要自动计算低频线的数量和位置，首先将掩码 mask 进行维度压缩（取其中一部分维度的元素，
#     这里取 [:, 0, 0, :, 0]，具体含义取决于掩码数据的结构和表示方式，可能是提取了一个关键的表示维度）
#     并转换为 torch.int8 类型，然后找到中心位置（通过取宽度维度的一半 cent = squeezed_mask.shape[1] // 2），
#     接着通过 argmin 函数从两边（左边通过翻转前半部分数据后找第一个非零元素，右边直接在后半部分找第一个非零元素）
#     找到低频线的边界位置，计算出对称的低频线数量（取两边最小值的 2 倍，但保证至少为 1，以保证有一定的低频线用于后续计算等情况）。
#     如果 num_low_frequencies 有指定值，则直接根据掩码的形状创建一个对应大小且值为指定低频线数量的张量。
#     最后根据掩码的宽度维度和计算得到的低频线数量计算填充量 pad（通过简单的数学计算使得在后续处理中数据的布局符合要求），
#     并将 pad 和 num_low_frequencies_tensor 转换为 torch.long 类型后返回，
#     这两个返回值在后续对 k - 空间数据进行掩码中心处理等操作时会用到。
#     '''
#     def get_pad_and_num_low_freqs(
#         self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if num_low_frequencies is None or num_low_frequencies == 0:
#             # get low frequency line locations and mask them out
#             squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
#             cent = squeezed_mask.shape[1] // 2
#             # running argmin returns the first non-zero
#             left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
#             right = torch.argmin(squeezed_mask[:, cent:], dim=1)
#             num_low_frequencies_tensor = torch.max(
#                 2 * torch.min(left, right), torch.ones_like(left)
#             )  # force a symmetric center unless 1
#         else:
#             num_low_frequencies_tensor = num_low_frequencies * torch.ones(
#                 mask.shape[0], dtype=mask.dtype, device=mask.device
#             )

#         pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

#         return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)
#     '''
#     在前向传播方法中，首先判断 self.mask_center 属性，如果为 True，则调用get_pad_and_num_low_freqs方法
#     根据掩码 mask 和可选的低频线数量参数 num_low_frequencies 计算得到填充量 pad 和
#     实际低频线数量 num_low_freqs，然后使用 transforms.batched_mask_center 函数
#     （来自 fastmri 库的 transforms 模块，应该是用于对批量的 k - 空间数据进行中心掩码处理的函数）
#     按照计算得到的参数对输入的 masked_kspace 进行掩码处理，模拟在 k - 空间中心区域的一些特定处理情况
#     （比如去除或调整中心部分数据，具体取决于函数功能和应用场景）。接着，将经过掩码处理（如果有进行掩码操作的话）的
#     masked_kspace 通过 fastmri.ifft2c 函数（假设是 fastmri 库中用于进行二维逆快速傅里叶变换的函数，
#     将k-空间数据转换到图像空间）转换到图像空间，再调用 chans_to_batch_dim 方法将通道维度合并到批量维度，
#     并记录原始的批量大小 batches，得到 images 张量。最后，将 images 传入 norm_unet（之前初始化的归一化 U - Net 实例）
#     进行处理，再通过 batch_chans_to_chan_dim 方法恢复维度结构，
#     然后调用divide_root_sum_of_squares方法进行均方根归一化操作，得到最终估计的敏感度值并返回，
#     这个返回的敏感度张量后续可以用于其他与 MRI 重建等相关的计算中。
#     '''
#     def forward(
#         self,
#         masked_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         num_low_frequencies: Optional[int] = None,
#     ) -> torch.Tensor:
#         if self.mask_center:
#             pad, num_low_freqs = self.get_pad_and_num_low_freqs(
#                 mask, num_low_frequencies
#             )
#             masked_kspace = transforms.batched_mask_center(
#                 masked_kspace, pad, pad + num_low_freqs
#             )

#         # convert to image space
#         images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

#         # estimate sensitivities
#         return self.divide_root_sum_of_squares(
#             self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
#         )


# class VarNet(nn.Module):
#     """
#     A full variational network model.

#     This model applies a combination of soft data consistency with a U-Net
#     regularizer. To use non-U-Net regularizers, use VarNetBlock.
#     """

#     def __init__(
#         self,
#         num_cascades: int = 12,
#         sens_chans: int = 8,
#         sens_pools: int = 4,
#         chans: int = 18,
#         pools: int = 4,
#         mask_center: bool = True,
#     ):
#         """
#         Args:
#             num_cascades: Number of cascades (i.e., layers) for variational
#                 network.
#             sens_chans: Number of channels for sensitivity map U-Net.
#             sens_pools Number of downsampling and upsampling layers for
#                 sensitivity map U-Net.
#             chans: Number of channels for cascade U-Net.
#             pools: Number of downsampling and upsampling layers for cascade
#                 U-Net.
#             mask_center: Whether to mask center of k-space for sensitivity map
#                 calculation.
#         """
#         super().__init__()

#         self.sens_net = SensitivityModel(
#             chans=sens_chans,
#             num_pools=sens_pools,
#             mask_center=mask_center,
#         )
#         self.cascades = nn.ModuleList(
#             [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
#         )
#     '''
#     forward 方法功能概述及流程解读（续）：
#     在前向传播过程中，首先通过调用 self.sens_net（即之前在初始化中构建的 SensitivityModel 实例）的 forward 方法，
#     传入 masked_kspace、mask 和 num_low_frequencies 参数，来获取敏感度图 sens_maps，
#     这一步是基于输入的掩码k-空间数据等信息计算出对应的线圈敏感度信息，对于后续在变分网络中的数据一致性处理等操作很关键。
#     然后创建一个变量 kspace_pred，并初始化为输入的 masked_kspace 的克隆，
#     它将在后续的循环迭代过程中不断更新，逐步逼近最终预测的k-空间数据。
#     接着进入一个循环，遍历 self.cascades（这是一个 nn.ModuleList，其中包含了多个VarNetBlock实例，
#     数量由 num_cascades 参数指定），对于每个 VarNetBlock 实例（也就是每一层的变分网络模块），
#     调用其 forward 方法，将当前的 kspace_pred、原始的 masked_kspace、mask 以及计算得到的 sens_maps 作为参数传入，
#     通过每层的计算不断更新 kspace_pred 的值，使其朝着更符合数据一致性和期望输出的方向调整，
#     每一层都在结合当前的预测、原始数据以及敏感度信息进行软数据一致性和正则化操作
#     （具体由 VarNetBlock 的逻辑实现，后面会详细解读）。
#     最后，经过多层迭代后，对最终的 kspace_pred 进行一系列操作来得到模型的输出结果。
#     先通过 fastmri.ifft2c 函数将 k-空间数据（kspace_pred）转换为图像空间（进行二维逆快速傅里叶变换），
#     然后使用 fastmri.complex_abs 函数获取复数的模（也就是图像空间中对应的数据幅度信息），
#     再调用 fastmri.rss 函数（可能是计算均方根之类的操作，用于对多通道或者多线圈等情况下的数据进行整合等处理，
#     具体功能取决于 fastmri 库中该函数的定义）在指定维度（这里是维度 1）上进行操作，最终得到的结果就是整个 VarNet 模型的输出，
#     通常可以理解为重建后的 MRI 图像相关的数据表示，返回给调用者用于后续的损失计算、评估等操作。
#     '''
#     def forward(
#         self,
#         masked_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         num_low_frequencies: Optional[int] = None,
#     ) -> torch.Tensor:
#         sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
#         kspace_pred = masked_kspace.clone()

#         for cascade in self.cascades:
#             kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

#         return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

# '''
# 类功能概述：VarNetBlock类定义了变分网络中的一个基本模块，它将软数据一致性（Soft Data Consistency）
# 操作与作为正则化器的输入模型（通过构造函数传入）相结合，多个这样的模块可以堆叠起来构成完整的变分网络
# （如前面 VarNet 类中通过 nn.ModuleList 堆叠了多个 VarNetBlock），在每一层中对输入的k-空间数据进行处理，
# 逐步优化预测结果，使其既符合数据一致性要求又能通过正则化约束避免过拟合等问题，更好地完成 MRI 图像重建等任务。
# '''
# class VarNetBlock(nn.Module):
#     """
#     Model block for end-to-end variational network.

#     This model applies a combination of soft data consistency with the input
#     model as a regularizer. A series of these blocks can be stacked to form
#     the full variational network.
#     """
#     '''
#     在初始化方法中，首先调用父类（nn.Module）的初始化方法 super().__init__() 确保父类相关的初始化逻辑得以执行
#     （比如注册模型参数等操作）。然后将传入的 model（应该是一个具有正则化功能的神经网络模型，例如 NormUnet 的实例，
#     在 VarNet 类的初始化中构造 VarNetBlock 时传入的就是 NormUnet 的实例）赋值给 self.model，
#     用于后续在该模块中进行正则化相关的操作。接着创建一个可学习的参数 self.dc_weight，初始化为值为 1 的张量（形状为 (1,)），
#     这个参数用于控制软数据一致性操作中的权重，在训练过程中会根据梯度下降等优化算法进行调整，
#     以平衡数据一致性和正则化两方面在模型更新中的作用。
#     '''
#     def __init__(self, model: nn.Module):
#         """
#         Args:
#             model: Module for "regularization" component of variational
#                 network.
#         """
#         super().__init__()

#         self.model = model
#         self.dc_weight = nn.Parameter(torch.ones(1))
#     '''
#     数据变换相关辅助方法解读:
#     此方法用于将输入张量 x（通常可以理解为与图像空间相关的数据表示，但在处理流程中可能涉及不同空间的数据转换等情况）
#     结合敏感度图 sens_maps 进行操作，先通过 fastmri.complex_mul 函数（应该是对复数数据进行逐元素相乘的操作，
#     按照复数乘法规则将 x 和 sens_maps 对应元素相乘），然后再使用 fastmri.fft2c 函数（二维快速傅里叶变换，
#     将相乘后的结果从图像空间转换回 k-空间），最终返回变换后的数据，实现了一种基于敏感度图将数据从图像空间转换回
#     k-空间并结合敏感度信息的操作，在变分网络的数据处理流程中，用于在不同空间之间进行转换和整合信息，
#     以便后续进行数据一致性等计算。
#     '''
#     def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

#     '''
#     该方法与 sens_expand 相对应，用于对输入张量 x（这里的 x 同样是在变分网络处理流程中的数据，可能处于不同空间状态）
#     结合敏感度图 sens_maps 进行另一种转换操作。首先通过 fastmri.ifft2c 函数将x从k-空间转换到图像空间
#     （进行二维逆快速傅里叶变换），然后使用 fastmri.complex_conj 函数获取 sens_maps 的共轭复数
#     （在复数运算中，与共轭复数相乘等操作常用于一些计算和处理，比如这里后续的计算），
#     接着通过 fastmri.complex_mul 函数将逆变换后的 x 与 sens_maps 的共轭复数逐元素相乘，
#     最后在维度 1 上进行求和（通过 sum(dim=1, keepdim=True)，keepdim=True 表示保持求和后的维度数量不变，
#     只是将维度 1 的大小变为 1，这样可以保持数据结构的一致性以便后续操作），返回得到经过这种基于敏感度图变换和维度缩减后的结果，
#     在整个变分网络模块的计算流程中，起到了从k-空间到图像空间的转换以及根据敏感度图进行信息整合和维度调整的作用，
#     便于后续与正则化模型等操作进行衔接。
#     '''
#     def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         return fastmri.complex_mul(
#             fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
#         ).sum(dim=1, keepdim=True)
#     '''
#     前向传播方法 forward 解读:
#     在前向传播方法中，首先创建一个全零张量 zero，其形状为 (1, 1, 1, 1, 1)，并将其移动到与 current_kspace 相同的设备
#     （如 GPU 或 CPU）上，这个全零张量用于后续构建软数据一致性操作中的替换值（在掩码为 False 的位置进行替换）。
#     接着构建软数据一致性项 soft_dc，通过 torch.where 函数根据掩码 mask 来确定计算方式，如果掩码对应位置为 True，
#     则计算 current_kspace - ref_kspace（这里 current_kspace 可以理解为当前层变分网络模块的输入k-空间数据预测值，
#     ref_kspace 通常是原始的输入掩码 k - 空间数据等参考值，它们的差值表示当前预测与原始数据的差异情况），
#     如果掩码对应位置为 False，则使用之前创建的全零张量 zero 进行替换，最后将这个根据掩码确定的差值结果乘以可学习的权重
#     self.dc_weight，这样就得到了软数据一致性项，它会根据掩码情况以及权重动态调整对当前预测数据与原始数据一致性的约束强度，
#     在训练过程中通过调整权重来平衡数据一致性和其他正则化等因素的影响。
#     然后构建模型正则化项 model_term，先通过调用 self.sens_reduce 方法对 current_kspace 和 sens_maps 进行处理，
#     将 k - 空间数据转换到图像空间并结合敏感度图进行信息整合和维度调整，得到的结果再传入self.model
#     （之前初始化传入的正则化模型，例如 NormUnet）进行正则化相关的特征提取等操作，
#     最后通过调用self.sens_expand方法将经过正则化处理后的数据再转换回 k-空间并结合敏感度图进行扩展，
#     得到模型正则化项，这个项体现了正则化模型对数据特征的调整和约束作用，有助于防止过拟合以及优化模型的重建性能。
#     最后，返回current_kspace - soft_dc - model_term，即将当前层的输入k-空间数据预测值减去软数据一致性项和模型正则化项，
#     通过这样的计算更新当前的预测值，使其在满足数据一致性的同时，受到正则化的约束，逐步向更好的重建结果逼近，
#     每一层 VarNetBlock 都进行这样的操作，经过多层堆叠后（如在 VarNet 类中多个 VarNetBlock 依次作用），
#     最终得到整个变分网络的输出结果，也就是重建后的MRI图像相关的数据表示（如前面VarNet类的forward方法中最后返回的结果形式）。
#     '''
#     def forward(    
#         self,
#         current_kspace: torch.Tensor,
#         ref_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor,
#     ) -> torch.Tensor:
#         zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
#         soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
#         model_term = self.sens_expand(
#             self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
#         )

#         return current_kspace - soft_dc - model_term
# '''
# 综上所述，这段代码整体构建了一个基于变分网络（VarNet）的 MRI 图像重建模型结构，
# 包含了不同的模块（NormUnet、SensitivityModel、VarNetBlock 等），
# 每个模块都有其特定的数据处理、特征提取以及在变分网络整体架构中承担的角色（如归一化、敏感度估计、软数据一致性与正则化结合等），
# 通过层层作用和协同工作来实现从输入的掩码 k - 空间数据到最终 MRI 图像重建结果的转换和优化过程。
# '''
# '''
# 相关问题：
# 除了U-Net，还有哪些常用的神经网络结构可以作为正则化器?
# 如何调整VarNet模型中的参数以获得更好的性能?
# 分享一些关于VarNet模型的应用案例
# '''