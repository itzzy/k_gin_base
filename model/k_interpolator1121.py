# Acknowledgement
# This part of code is developed based on the repository MAE: https://github.com/facebookresearch/mae.

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, PatchEmbed,Mlp
from timm.models.layers import DropPath
from utils.model_related import get_2d_sincos_pos_embed
from utils import ifft2c


class Attention(nn.Module):
    '''
    dim: 输入特征的维度。
    num_heads: 注意力头的数量，默认为 8。
    qkv_bias: 是否在查询（Q）、键（K）、值（V）的线性变换中添加偏置，默认为 False。
    qk_scale: 缩放因子，用于 Q 和 K 的点积，默认为 None，在这种情况下，缩放因子是 head_dim ** -0.5。
    attn_drop: 注意力权重的 dropout 概率，默认为 0。
    proj_drop: 输出投影的 dropout 概率，默认为 0。
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # 计算每个注意力头的维度。
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # 计算缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 创建一个 dropout 层用于注意力权重
        self.attn_drop = nn.Dropout(attn_drop)
        # 创建一个线性层用于将注意力输出投影到原始维度
        self.proj = nn.Linear(dim, dim)
        # 创建一个 dropout 层用于输出投影
        self.proj_drop = nn.Dropout(proj_drop)
    # 定义了注意力机制的前向传播过程
    def forward(self, kv, x):
        # B, N, C 分别代表批次大小、序列长度和特征维度
        B, N, C = x.shape
        #print('kv, x', kv.shape, x.shape) kv, x torch.Size([1, 3457, 512]) torch.Size([1, 3457, 512])
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #k 和 v 是通过 kv 参数计算得到的键和值张量。
        k = kv.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = k
        #q 是通过输入 x 计算得到的查询张量
        q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 是通过查询和键的点积计算得到的注意力权重，然后乘以缩放因子。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 应用 dropout
        attn = self.attn_drop(attn)
        # (attn @ v) 计算加权的值张量  重新排列和重塑输出张量。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 应用线性层进行输出投影。
        x = self.proj(x)
        # 应用dropout
        x = self.proj_drop(x)
        
        # x: (B, N, C)
        return x

class DecoderTrans(nn.Module):
    #def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer):
    '''
    dim: 输入特征的维度。
    num_heads: 注意力头的数量。
    mlp_ratio: MLP（多层感知机）的隐藏层维度与输入维度的比例，默认为 4。
    qkv_bias: 是否在查询（Q）、键（K）、值（V）的线性变换中添加偏置，默认为 False。
    qk_scale: 缩放因子，用于 Q 和 K 的点积，默认为 None。
    drop: MLP 的 dropout 概率，默认为 0。
    attn_drop: 注意力权重的 dropout 概率，默认为 0。
    drop_path: Stochastic Depth 的 dropout 概率，默认为 0。
    norm_layer: 归一化层，默认为: nn.LayerNorm。
    act_layer: 激活函数，默认为: nn.GELU。
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        # 创建一个归一化层实例 self.norm1，它将应用于输入 x
        self.norm1 = norm_layer(dim)
        # 创建一个 Attention 实例 self.attn，它将用于计算注意力权重
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        '''
        DropPath 是一种正则化技术，用于随机丢弃（Dropout）Transformer 模型中的某些路径，以防止过拟合。
        在 PyTorch 中，DropPath 通常不是内置的，因此你需要从其他库中导入它。在 timm 库中，
        DropPath 是一个可用的类，它实现了随机路径丢弃的功能。
        
        根据 drop_path 的值创建一个 DropPath 实例或 nn.Identity。如果 drop_path 大于 0，
        则使用 DropPath 来实现随机路径丢弃；否则，使用 nn.Identity 作为恒等函数。
        '''
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 创建另一个归一化层实例 self.norm2，它将应用于 self.mlp 的输出
        self.norm2 = norm_layer(dim)
        # 计算 MLP 层的隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 创建一个 Mlp 实例 self.mlp，它将用于实现 MLP 层。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, kv, x):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 将输入 x 与 self.attn 的输出相加，其中 self.attn 的输出是通过 kv 和 x 计算得到的注意力权重。self.drop_path 用于应用随机路径丢弃。
        x = x + self.drop_path(self.attn(kv,x))
        # 将输入 x 与 self.mlp 的输出相加，其中 self.mlp 的输出是通过 self.norm2 归一化后的 x 计算得到的。self.drop_path 用于应用随机路径丢弃。
        x = x + self.drop_path(self.mlp(self.norm2(x)))        
        return x

# 用于图像重建的深度学习模型  定义了一个复杂的深度学习模型，
# 它结合了卷积层、Transformer编码器和解码器、位置嵌入、掩码标记等组件，用于图像重建任务。
# 模型的设计考虑了多种调整策略，以适应不同的图像重建需求。 
class KInterpolator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从传入的配置对象中提取 KInterpolator 相关的配置参数
        config = config.KInterpolator
        # 初始化模型的图像尺寸、输入通道数和嵌入维度
        self.img_size = config.img_size
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        # 从配置中提取深度学习模型的深度、注意力头的数量、解码器嵌入维度、解码器深度和解码器注意力头的数量。
        depth = config.depth
        num_heads = config.num_heads
        self.decoder_embed_dim = config.decoder_embed_dim
        decoder_depth = config.decoder_depth
        decoder_num_heads = config.decoder_num_heads
        mlp_ratio = config.mlp_ratio
        # 使用 eval 函数动态地评估配置中的 norm_layer 和 act_layer 字符串，以获取实际的归一化层和激活函数类。
        norm_layer = eval(config.norm_layer)
        act_layer = eval(config.act_layer)
        self.xt_y_tuning = config.xt_y_tuning
        self.yt_x_tuning = config.yt_x_tuning
        self.ref_repl_prior_denoiser = config.ref_repl_prior_denoiser
        self.post_tuning = True if self.xt_y_tuning or self.yt_x_tuning else False
        # 根据配置参数设置模型的特定调整选项
        self.xy_t_patch_tuning = config.xy_t_patch_tuning
        # 计算图像尺寸的补丁数量
        self.num_patches = self.img_size[0] * self.img_size[1]
        # 初始化一个可学习的类标记（class token），用于Transformer模型。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # 初始化位置嵌入参数，用于为每个补丁添加位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # 生成一个随机的偏置向量 B，并将其注册为缓冲区，这意味着它不会被优化器更新。
        B = torch.randn((1, 1, self.embed_dim//2), dtype=torch.float32)
        self.register_buffer('B', B)
        # 定义一个卷积层，用于将输入图像的通道数从 img_size[2]*2 转换为嵌入维度 embed_dim
        self.patch_embed = nn.Conv2d(self.img_size[2]*2, self.embed_dim, kernel_size=(1, 1))

        # 创建一个模块列表，包含 depth 个 Block 实例，每个 Block 是一个Transformer编码器层
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])   
        #  创建一个归一化层，用于Transformer编码器的输出
        self.norm = norm_layer(self.embed_dim)
        # 定义一个线性层，用于将编码器的嵌入维度映射到解码器的嵌入维度
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        # 初始化一个可学习的掩码标记（mask token），用于掩码补丁的预测
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        # 初始化解码器的位置嵌入参数
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # 创建一个模块列表，包含 decoder_depth 个 DecoderTrans 实例，每个 DecoderTrans 是一个Transformer解码器层。
        self.decoder_blocks = nn.ModuleList([
            DecoderTrans(self.decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(decoder_depth)])
        # 创建一个归一化层，用于Transformer解码器的输出
        self.decoder_norm = norm_layer(self.decoder_embed_dim)
        # 定义一个线性层，用于将解码器的嵌入维度映射回图像尺寸
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.img_size[2]*2, bias=True)
        # 如果启用了 yt_x_tuning，则会添加额外的模块来处理特定的调整
        if self.yt_x_tuning:
            self.yt_x_num_patches = self.num_patches
            self.yt_x_pos_embed = nn.Parameter(torch.zeros(1, self.yt_x_num_patches, config.yt_x_embed_dim),
                                               requires_grad=False)
            self.yt_x_patch_embed = nn.Conv2d(self.img_size[2] * 2, config.yt_x_embed_dim, kernel_size=(1, 1))
            self.yt_x_blocks = nn.ModuleList([
                Block(config.yt_x_embed_dim, config.yt_x_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                      act_layer=act_layer) for i in range(config.yt_x_depth)])
            self.yt_x_norm = norm_layer(config.yt_x_embed_dim)
            self.yt_x_pred = nn.Linear(config.yt_x_embed_dim, self.img_size[2] * 2,
                                       bias=True)
        # 如果启用了 xt_y_tuning，则会添加额外的模块来处理特定的调整。
        if self.xt_y_tuning:
            self.xt_y_num_patches = self.img_size[0] * self.img_size[2]
            self.xt_y_pos_embed = nn.Parameter(torch.zeros(1, self.xt_y_num_patches, config.xt_y_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.xt_y_patch_embed = nn.Conv2d(self.img_size[1]*2, config.xt_y_embed_dim, kernel_size=(1, 1))

            self.xt_y_blocks = nn.ModuleList([
                Block(config.xt_y_embed_dim, config.xt_y_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
                for i in range(config.xt_y_depth)])
            self.xt_y_norm = norm_layer(config.xt_y_embed_dim)
            self.xt_y_pred = nn.Linear(config.xt_y_embed_dim, self.img_size[1]*2, bias=True)
        # 如果启用了 xy_t_patch_tuning，则会添加额外的模块来处理特定的调整
        if self.xy_t_patch_tuning:
            self.xy_t_patch_embed = PatchEmbed(self.img_size[-1:0:-1], config.patch_size, self.img_size[0]*2, config.xy_t_patch_embed_dim)
            self.xy_t_patch_pos_embed = nn.Parameter(torch.zeros(1, self.xy_t_patch_embed.num_patches, config.xy_t_patch_embed_dim), requires_grad=False)
            self.xy_t_patch_blocks = nn.ModuleList([
                Block(config.xy_t_patch_embed_dim, config.xy_t_patch_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
                for i in range(config.xy_t_patch_depth)])
            self.xy_t_patch_norm = norm_layer(config.xy_t_patch_embed_dim)
            self.xy_t_patch_pred = nn.Linear(config.xy_t_patch_embed_dim, config.patch_size**2*self.img_size[0]*2, bias=True)
        # 创建一个归一化层，用于最终的全连接层
        self.fc_norm = norm_layer(32)
    '''
    初始化模型参数，包括位置编码、分类标记、掩码标记等。
    位置编码用于告诉模型每个 patch 在图像中的位置信息，有助于模型学习空间关系。
    分类标记用于表示图像的类别信息。
    掩码标记用于表示哪些 patch 被掩盖。
    _init_weights 函数使用 Xavier 均匀初始化来初始化 nn.Linear 层的权重，并使用常数初始化来初始化 nn.LayerNorm 层的偏置和权重。
    '''
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #使用 get_2d_sincos_pos_embed 函数生成正弦余弦位置编码，并将它们复制到模型参数中。
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.img_size[0], self.img_size[1], cls_token=True)
        self.pos_embed.data.copy_(torch.tensor(pos_embed).unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.img_size[0], self.img_size[1], cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.tensor(decoder_pos_embed).unsqueeze(0))

        if self.xt_y_tuning:
            xt_y_pos_embed = get_2d_sincos_pos_embed(self.xt_y_pos_embed.shape[-1], self.img_size[0], self.img_size[2], cls_token=False)
            self.xt_y_pos_embed.data.copy_(torch.tensor(xt_y_pos_embed).unsqueeze(0))

        if self.yt_x_tuning:
            yt_x_pos_embed = get_2d_sincos_pos_embed(self.yt_x_pos_embed.shape[-1], self.img_size[0], self.img_size[1], cls_token=False)
            self.yt_x_pos_embed.data.copy_(torch.tensor(yt_x_pos_embed).unsqueeze(0))

        if self.xy_t_patch_tuning:
            xy_t_patch_pos_embed = get_2d_sincos_pos_embed(self.xy_t_patch_pos_embed.shape[-1], self.xy_t_patch_embed.grid_size[0], self.xy_t_patch_embed.grid_size[1], cls_token=False)
            self.xy_t_patch_pos_embed.data.copy_(torch.tensor(xy_t_patch_pos_embed).unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # 使用 torch.nn.init.normal_ 函数对分类标记和掩码标记进行随机初始化
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # 使用 self.apply(self._init_weights) 递归地调用 _init_weights 函数来初始化模型中的 nn.Linear 和 nn.LayerNorm 层。
        self.apply(self._init_weights)

    '''
    初始化 nn.Linear 和 nn.LayerNorm 层的参数。
    Xavier 均匀初始化可以帮助防止梯度消失或爆炸，确保训练的稳定性。
    常数初始化通常用于初始化偏置和权重，可以加速训练过程。
    '''
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            # 如果 m 是 nn.Linear 层，则使用 Xavier 均匀初始化来初始化权重，并使用常数初始化来初始化偏置。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # 如果 m 是 nn.LayerNorm 层，则使用常数初始化来初始化偏置和权重。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    '''
    编码 k-space 数据，提取特征。
    kspace: k-space 数据，形状为 (B, C, H, W)。
    mask: 掩码，形状为 (B, H * W)，指示哪些 patch 被掩盖。
    '''
    def encoder(self, kspace, mask):
        b, c, h, w = kspace.shape
        # 使用 self.patch_embed 将 k-space 数据分割成 patch，并进行 embedding。
        kspace = self.patch_embed(kspace)
        # 将 patch 扁平化为 (B, N, C) 的形状。
        kspace = kspace.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #print('k, m', kspace.shape, mask.shape) 添加位置编码。
        kspace = kspace + self.pos_embed[:, 1:, :]
        # 对未被掩盖的 patch 进行重新排列
        kspace = kspace[mask > 0, :].reshape(b, -1, self.embed_dim)
        ids_shuffle = torch.argsort(mask, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # 将分类标记添加到 patch 中,使用 Transformer 模块处理 patch，提取特征。
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(kspace.shape[0], -1, -1)
        kspace = torch.cat((cls_tokens, kspace), dim=1)
        
#        imgadd = torch.mean(tensor, dim=2, keepdim=True)
#        imgadd = torch.cat((mean_of_third_dim, tensor), dim=2)

        # apply Transformer blocks
        for blk in self.blocks:
            kspace = blk(kspace)
        # 使用 self.norm 对特征进行归一化
        kspace = self.norm(kspace)
        # kspace: 编码后的特征，形状为 (B, N + 1, C).ids_restore: 用于恢复 patch 的原始顺序，形状为 (B, N)
        return kspace, ids_restore
    
    # 将编码后的特征重新组装成图像数据。
    # 输入: x: 编码后的特征，形状为 (N, L, patch_size**2 *3)。
    def unpatchify_xy_t(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # 重新组合特征，形成 (N, H, W, C) 的形状。
        p = self.xy_t_patch_embed.patch_size[0]
        h, w = self.xy_t_patch_embed.grid_size[0], self.xy_t_patch_embed.grid_size[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.img_size[0]*2))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.img_size[0], 2, h * p, w * p))
        # 重新组装后的图像数据，形状为 (N, C, H, W)。
        imgs = torch.einsum('btchw->bthwc', imgs)
        return imgs
    
    
    '''
    解码编码后的特征，重建 k-space 数据
    kv: 编码器输出的键值对，形状为 (B, N + 1, C)。
    q: 编码器输出的查询，形状为 (B, N + 1, C)。
    ids_restore: 用于恢复 patch 的原始顺序，形状为 (B, N)。
    mask: 掩码，形状为 (B, N + 1)，指示哪些 patch 被掩盖。
    '''
    def decoder(self, kv, q, ids_restore, mask):
        #print('kv, q', kv.shape, q.shape) #kv, q torch.Size([1, 3457, 512]) torch.Size([1, 919, 512]) 
        # 使用 self.decoder_embed 将查询嵌入到解码器
        kspace = self.decoder_embed(q)

        mask_tokens = self.mask_token.repeat(kspace.shape[0], ids_restore.shape[1] + 1 - kspace.shape[1], 1)
        kspace_full = torch.cat([kspace[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # 使用 ids_restore 恢复 patch 的原始顺序
        kspace_full = torch.gather(kspace_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, kspace.shape[2]))  # unshuffle
        kspace = torch.cat([kspace[:, :1, :], kspace_full], dim=1)  # append cls token

        # add pos embed 添加位置编码。
        kspace = kspace + self.decoder_pos_embed
        #print('kspace', kspace.shape) kspace torch.Size([1, 3457, 512])
        # apply Transformer blocks
        # 使用 Transformer 模块处理 patch，重建 k-space 数据。
        for blk in self.decoder_blocks:
            kspace = blk(kv, kspace)
        # 使用 self.decoder_norm 对重建后的数据进行归一化。
        kspace = self.decoder_norm(kspace)

        latent_decoder = kspace[:, 1:, :][mask==0, :].reshape(kspace.shape[0], -1, self.decoder_embed_dim)

        # predictor projection 使用 self.decoder_pred 对重建后的数据进行预测。
        kspace = self.decoder_pred(kspace)

        # remove cls token
        kspace = kspace[:, 1:, :]
        # 重建后的 k-space 数据，形状为 (B, N, C)。 解码器输出的潜在特征，形状为 (B, N, C)。
        return kspace, latent_decoder
    # 处理 k-space 数据，进行不同维度的转换和处理。 k-space 数据，形状为 (B, T, H, W, C),分别代表批次大小、时间帧数量、图像高度、图像宽度和通道数量。
    def xt_y(self, kspace):
        b, t, h, w, c = kspace.shape
        '''
        对 k-space 数据进行维度变换。torch.einsum 函数使用爱因斯坦求和约定对张量进行操作，
        'bthwc->bwcth' 表示将输入张量 kspace 的维度从 (b, t, h, w, c) 变换为 (b, w, c, t, h)。
        flatten(1,2) 函数将维度 1 和 2 扁平化为一个维度，即 (b, w*c, t, h)
        '''
        kspace = torch.einsum('bthwc->bwcth', kspace).flatten(1,2)
        '''
        将处理后的 k-space 数据输入到 xt_y_patch_embed 模块中进行处理。
        该模块可能是用于将 k-space 数据分割成 patch 并进行 embedding 的一个模块。
        '''
        kspace = self.xt_y_patch_embed(kspace)
        '''
        对数据进行再次维度变换。flatten(2) 函数将维度 2 扁平化为一个维度，即 (b, w*c*t, h)。
        transpose(1, 2) 函数将维度 1 和 2 进行交换，得到 (b, h, w*c*t) 的形状。
        '''
        kspace = kspace.flatten(2).transpose(1, 2)  # BCN -> BNC
        # 将处理后的 k-space 数据加上 xt_y_pos_embed，该参数可能是位置编码，用于提供位置信息。
        kspace = kspace + self.xt_y_pos_embed
        '''
        循环遍历 self.xt_y_blocks 中的每个模块。self.xt_y_blocks 可能是一个包含多个 Transformer 模块的列表，
        用于对 k-space 数据进行特征提取和处理。
        '''
        for blk in self.xt_y_blocks:
            # 将 k-space 数据输入到当前循环中的 Transformer 模块 blk 中进行处理
            kspace = blk(kspace)
        kspace = self.xt_y_norm(kspace)
        kspace = self.xt_y_pred(kspace)
        return kspace.reshape((b, t, h, w, 2))
    # 这个函数处理对 k 空间数据进行的一种特定类型的调整
    def x_yt(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->btwch', kspace).flatten(1,3)
        kspace = self.x_yt_patch_embed(kspace)
        kspace = kspace.transpose(1, 2)  # BCN -> BNC
        kspace = kspace + self.x_yt_pos_embed
        for blk in self.x_yt_blocks:
            kspace = blk(kspace)
        kspace = self.x_yt_norm(kspace)
        kspace = self.x_yt_pred(kspace)

        kspace = kspace.reshape((b, h, t, w, 2))
        return torch.einsum('bhtwc->bthwc', kspace)

    def yt_x(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->bhctw', kspace).flatten(1, 2)
        kspace = self.yt_x_patch_embed(kspace)
        kspace = kspace.flatten(2).transpose(1, 2)
        kspace = kspace + self.yt_x_pos_embed
        for blk in self.yt_x_blocks:
            kspace = blk(kspace)
        kspace = self.yt_x_norm(kspace)
        kspace = self.yt_x_pred(kspace)
        kspace = kspace.reshape((b, t, w, h, 2))

        return torch.einsum('btwhc->bthwc', kspace)

    def xy_t_patch(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->btchw', kspace).flatten(1, 2)
        kspace = self.xy_t_patch_embed(kspace)


        kspace = kspace + self.xy_t_patch_pos_embed
        for blk in self.xy_t_patch_blocks:
            kspace = blk(kspace)
        kspace = self.xy_t_patch_norm(kspace)
        kspace = self.xy_t_patch_pred(kspace)
        kspace = self.unpatchify_xy_t(kspace)
        return kspace.contiguous()
    
    def forward(self, img, mask):
        # size of input img and mask: [B, T, H, W]
        #print('k, m', img.shape, mask.shape) k, m torch.Size([1, 18, 192, 192]) torch.Size([1, 18, 192])
        
        # 预处理: 对输入图像和掩码进行预处理，包括计算参考图像 (img_0F) 和扩展掩码以匹配图像维度。  多通道合并
        img_0F = img.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)  # 计算参考图像，它是所有帧的平均值
        img_0F = img_0F.repeat(1, img.shape[1], 1, 1)  # 将参考图像复制到所有帧
        
        mask_0F = torch.ones(mask.shape[0], mask.shape[1], mask.shape[2])  # 创建一个全为 1 的掩码，表示所有位置都被观测到
        
        img_orig = torch.view_as_real(img)  # 将输入图像转换为实数形式
        mask_orig = mask[..., None, None].expand_as(img_orig)  # 将掩码扩展到与图像相同的维度
        
        img = torch.view_as_real(torch.einsum('bthw->btwh', img)).flatten(-2)  # 将图像数据转换为实数形式并扁平化
        img = torch.einsum('bhwt->bthw', img)  # 重新排列维度
        img_0F = torch.view_as_real(torch.einsum('bthw->btwh', img_0F)).flatten(-2)  # 将参考图像数据转换为实数形式并扁平化
        img_0F = torch.einsum('bhwt->bthw', img_0F)  # 重新排列维度
        b, h_2, t, w = img.shape  # 获取图像的维度信息

        mask = mask.flatten(1, -1)  # 将掩码扁平化
        mask_0F = mask_0F.flatten(1, -1)  # 将全为 1 的掩码扁平化
        
        # 编码: 使用 encoder 函数对参考图像和输入图像进行编码。这涉及将数据分割成补丁，应用 Transformer 块，并对输出进行归一化。
        kv, _ = self.encoder(img_0F, mask_0F)  # 对参考图像进行编码，得到键值对 (kv)
        q, ids_restore = self.encoder(img, mask)  # 对输入图像进行编码，得到查询 (q) 和用于恢复原始顺序的索引 (ids_restore)
        
        # 解码: 使用 decoder 函数对编码后的特征进行解码。这涉及对 Transformer 块进行应用以重建 k 空间数据。
        pred, latent_decoder = self.decoder(kv, q, ids_restore, mask)  # 使用解码器重建 k 空间数据，并得到预测结果 (pred) 和解码器输出的潜在特征 (latent_decoder)

        pred = pred.reshape((b, t, w, int(h_2/2), 2))  # 将预测结果重塑为 [B, T, W, H/2, 2] 的形状
        pred = torch.einsum('btwhc->bthwc', pred)  # 重新排列维度
        pred_list = [pred]  # 创建一个列表，用于存储不同调整阶段的预测结果

        pred_t = pred.clone()  # 复制预测结果
        if self.ref_repl_prior_denoiser: pred_t[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]  # 如果启用了参考替换先验去噪，则将掩码区域的值替换为原始图像的值
        
        # 调整: 然后将重建的 k 空间数据传递给调整函数 (xt_y, yt_x, xy_t_patch)，具体取决于模型的配置。
        if self.yt_x_tuning:
            pred_t = self.yt_x(pred_t) + pred_t  # 应用 yt_x 调整
            pred_list.append(pred_t)  # 将调整后的结果添加到列表中
        pred_t1 = pred_t.clone()  # 复制调整后的结果
        if self.ref_repl_prior_denoiser: pred_t1[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]  # 再次将掩码区域的值替换为原始图像的值
        
        if self.xt_y_tuning:
            pred_t1 = self.xt_y(pred_t1) + pred_t1  # 应用 xt_y 调整
            pred_list.append(pred_t1)  # 将调整后的结果添加到列表中
        pred_t2 = pred_t1.clone()  # 复制调整后的结果
        if self.ref_repl_prior_denoiser: pred_t2[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]  # 再次将掩码区域的值替换为原始图像的值
        
        if self.xy_t_patch_tuning:
            pred_t2 = self.xy_t_patch(pred_t2) + pred_t2  # 应用 xy_t_patch 调整
            pred_list.append(pred_t2)  # 将调整后的结果添加到列表中
        pred_t3 = pred_t2.clone()  # 复制调整后的结果
        if self.ref_repl_prior_denoiser: pred_t3[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]  # 再次将掩码区域的值替换为原始图像的值
        
        # 重建: 使用逆傅里叶变换 (ifft2c) 从最终的 k 空间数据重建图像。
        k_recon_complex = torch.view_as_complex(pred_t3)  # 将最终的预测结果转换为复数形式
        im_recon = ifft2c(k_recon_complex.to(torch.complex64))  # 使用逆傅里叶变换重建图像
        
        return pred_list, im_recon  # 返回所有调整阶段的预测结果列表和重建的图像

    # def forward(self, img, mask):
    #     # size of input img and mask: [B, T, H, W]
    #     #print('k, m', img.shape, mask.shape) k, m torch.Size([1, 18, 192, 192]) torch.Size([1, 18, 192])
    #     # 预处理: 对输入图像和掩码进行预处理，包括计算参考图像 (img_0F) 和扩展掩码以匹配图像维度。
    #     img_0F = img.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
    #     img_0F = img_0F.repeat(1, img.shape[1], 1, 1)
    #     #torch.mean(img, dim=1, keepdim=True).repeat(1, img.shape[1], 1, 1)
                
    #     mask_0F = torch.ones(mask.shape[0], mask.shape[1], mask.shape[2])
    #     #mask_0F = torch.ones(mask.shape[0], 1, mask.shape[2])
        
        
    #     img_orig = torch.view_as_real(img)       
    #     mask_orig = mask[..., None, None].expand_as(img_orig) #torch.Size([1, 18, 192, 192, 2])
        
    #     img = torch.view_as_real(torch.einsum('bthw->btwh', img)).flatten(-2)
    #     img = torch.einsum('bhwt->bthw', img)
    #     img_0F = torch.view_as_real(torch.einsum('bthw->btwh', img_0F)).flatten(-2)
    #     img_0F = torch.einsum('bhwt->bthw', img_0F)
    #     b, h_2, t, w = img.shape

    #     mask = mask.flatten(1, -1)
    #     mask_0F = mask_0F.flatten(1, -1) 
    #     # 编码: 使用 encoder 函数对参考图像和输入图像进行编码。这涉及将数据分割成补丁，应用 Transformer 块，并对输出进行归一化。
    #     kv, _ = self.encoder(img_0F, mask_0F)
    #     q, ids_restore = self.encoder(img, mask)
    #     # 解码: 使用 decoder 函数对编码后的特征进行解码。这涉及对 Transformer 块进行应用以重建 k 空间数据。
    #     pred, latent_decoder = self.decoder(kv, q, ids_restore, mask)

    #     pred = pred.reshape((b, t, w, int(h_2/2), 2))
    #     pred = torch.einsum('btwhc->bthwc', pred)
    #     pred_list = [pred]

    #     pred_t = pred.clone()
    #     if self.ref_repl_prior_denoiser: pred_t[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]
    #     # 调整: 然后将重建的 k 空间数据传递给调整函数 (xt_y, yt_x, xy_t_patch)，具体取决于模型的配置。
    #     if self.yt_x_tuning:
    #         pred_t = self.yt_x(pred_t) + pred_t
    #         pred_list.append(pred_t)
    #     pred_t1 = pred_t.clone()
    #     if self.ref_repl_prior_denoiser: pred_t1[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

    #     if self.xt_y_tuning:
    #         pred_t1 = self.xt_y(pred_t1) + pred_t1
    #         pred_list.append(pred_t1)
    #     pred_t2 = pred_t1.clone()
    #     if self.ref_repl_prior_denoiser: pred_t2[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

    #     if self.xy_t_patch_tuning:
    #         pred_t2 = self.xy_t_patch(pred_t2) + pred_t2
    #         pred_list.append(pred_t2)
    #     pred_t3 = pred_t2.clone()
    #     if self.ref_repl_prior_denoiser: pred_t3[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

    #     # 重建: 使用逆傅里叶变换 (ifft2c) 从最终的 k 空间数据重建图像。
    #     k_recon_complex = torch.view_as_complex(pred_t3)
    #     im_recon = ifft2c(k_recon_complex.to(torch.complex64))

    #     return pred_list, im_recon

'''
这段代码中的 `xt_y_tuning`, `yt_x_tuning`, 和 `xy_t_patch_tuning` 这三个参数代表着模型训练中是否启用三种不同的数据增强方法。

* **`xt_y_tuning`:**  这个参数控制着是否启用沿着时间轴和 Y 轴进行数据增强。具体来说，它会在解码器输出的基础上，使用 `self.xt_y()` 函数进行一次转换，这个转换会沿着时间轴和 Y 轴对数据进行处理，然后将处理结果加回到解码器输出中，以此来增强模型的学习能力。

* **`yt_x_tuning`:** 这个参数控制着是否启用沿着时间轴和 X 轴进行数据增强。类似于 `xt_y_tuning`，它会在解码器输出的基础上，使用 `self.yt_x()` 函数进行一次转换，这个转换会沿着时间轴和 X 轴对数据进行处理，然后将处理结果加回到解码器输出中。

* **`xy_t_patch_tuning`:** 这个参数控制着是否启用对空间和时间维度上的 patch 进行数据增强。它会在解码器输出的基础上，使用 `self.xy_t_patch()` 函数进行一次转换，这个转换会对空间和时间维度上的 patch 进行处理，然后将处理结果加回到解码器输出中。

简单来说，这三种数据增强方法分别对模型学习能力的不同方面进行增强，通过组合不同的数据增强方法，可以提高模型的泛化能力和鲁棒性。例如：

*  `xt_y_tuning` 可以增强模型对时间轴和 Y 轴方向上变化的敏感度。
*  `yt_x_tuning` 可以增强模型对时间轴和 X 轴方向上变化的敏感度。
*  `xy_t_patch_tuning` 可以增强模型对空间和时间维度上 patch 的特征提取能力。

在实际训练中，可以根据具体的数据集和任务选择是否启用这些数据增强方法，以及选择哪些数据增强方法组合。

'''

'''
代码中的'encoder`和`decoder’模块是如何协同工作的?它们分别负责处理哪些类型的 k-space 数据?
这段代码中的 `encoder` 和 `decoder` 模块协同工作，共同完成对 k-space 数据的编码、解码和重建过程。

**1. 编码器 (`encoder`)**

* 负责处理原始的 k-space 数据，提取其中的特征信息。
* 将 k-space 数据分割成 patch，并进行 embedding。
* 添加位置编码，告诉模型每个 patch 在图像中的位置信息。
* 使用 Transformer 模块处理 patch，提取特征。
* 对特征进行归一化。

**2. 解码器 (`decoder`)**

* 负责根据编码器提取的特征信息，重建 k-space 数据。
* 使用编码器输出的键值对 (kv) 和查询 (q) 进行解码。
* 使用 Transformer 模块处理 patch，重建 k-space 数据。
* 对重建后的数据进行归一化。
* 对重建后的数据进行预测，得到最终的 k-space 数据。

**协同工作机制:**

* 编码器将原始 k-space 数据转换为高维特征，这些特征包含了图像的重要信息。
* 解码器使用编码器提取的特征信息，重建原始的 k-space 数据。

**k-space 数据处理:**

* **编码器处理的 k-space 数据:**  原始的 k-space 数据，包含所有采样点信息。
* **解码器处理的 k-space 数据:** 编码器提取的特征信息，以及原始 k-space 数据中部分采样点的缺失信息。

**总结:**

编码器负责提取 k-space 数据中的特征信息，解码器负责使用这些特征信息重建 k-space 数据。它们协同工作，共同完成对 k-space 数据的编码、解码和重建过程。

'''

'''
重新组合特征，形成 (N, H, W, C) 的形状。  N, H, W, 含义是？  k-space 数据，形状为 (B, T, H, W, C)，B, T, H, W, C含义是？
好的，我来解释一下。
1. 重新组合特征，形成 (N, H, W, C) 的形状

- N:  样本数量 (Number of samples)，表示你拥有多少个图像数据。
- H: 图像高度 (Height)，表示每个图像的高度像素数量。
- W: 图像宽度 (Width)，表示每个图像的宽度像素数量。
- C:  通道数量 (Number of channels)，表示每个像素包含多少个通道信息，例如 RGB 图像有 3 个通道，单通道灰度图像有 1 个通道。

2. k-space 数据，形状为 (B, T, H, W, C)

- B: 批次大小 (Batch size)，表示一次处理的样本数量。
- T: 时间帧数量 (Number of time frames)，表示一个完整的图像序列包含多少帧图像。
- H: 图像高度 (Height)，表示每个图像的高度像素数量。
- W: 图像宽度 (Width)，表示每个图像的宽度像素数量。
- C: 通道数量 (Number of channels)，表示每个像素包含多少个通道信息。

简而言之， (N, H, W, C) 是单个图像数据的形状，而 (B, T, H, W, C) 则是包含多个图像数据的批次数据的形状，其中 T 表示时间维度上的帧数。

希望以上解释能够帮助你理解！ 
'''