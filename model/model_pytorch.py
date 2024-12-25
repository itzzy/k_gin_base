import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import model.kspace_pytorch as cl

'''
这个函数定义了一个带泄漏的修正线性单元（LeakyReLU）激活函数，斜率参数设置为 0.01，并且设置 inplace=True，
意味着在原地进行操作，节省内存空间，直接修改输入张量而不需要额外复制一份，函数返回 nn.LeakyReLU 激活函数的实例，
后续可以在神经网络模块中用于非线性变换。
'''
def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

'''
该函数定义了常见的修正线性单元（ReLU）激活函数，同样设置 inplace=True 以实现原地操作，返回 nn.ReLU 激活函数的实例，
用于给神经网络添加非线性特性，将输入张量中小于 0 的值置为 0，大于等于 0 的值保持不变。
'''
def relu():
    return nn.ReLU(inplace=True)

'''
函数参数说明：
n_ch：输入通道数，指定输入数据的通道数量。
nd：卷积层数，决定了整个卷积块中包含的卷积操作次数。
nf：卷积核数量（滤波器数量），默认值为 32，表示每个卷积层中卷积核的数量，决定了输出特征图的通道数变化情况。
ks：卷积核大小，默认是 3，确定卷积核的尺寸。
dilation：空洞卷积的扩张率，用于控制卷积核元素之间的间隔，默认为 1，即普通卷积，大于 1 时为空洞卷积。
bn：是否使用批归一化（Batch Normalization），布尔值，若为 True，则在卷积层间添加批归一化操作，有助于加速训练和提高模型泛化能力。
nl：激活函数类型，默认为 'lrelu'，可以选择 'relu' 或 'lrelu'，决定使用哪种激活函数进行非线性变换。
conv_dim：卷积维度，取值为 2 或 3，用于确定是进行二维卷积（2D）还是三维卷积（3D），默认是 2。
n_out：输出通道数，若为 None，则默认输出通道数和输入通道数 n_ch 相同。
'''
def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    # 首先根据 conv_dim 的值确定是使用二维卷积 nn.Conv2d 还是三维卷积 nn.Conv3d，并将对应的卷积操作类赋值给 conv 变量。
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    # 接着处理输出通道数 n_out，若未传入具体值，则将其设置为与输入通道数 n_ch 相等。
    if not n_out:
        n_out = n_ch

    # dilated convolution
    # 对于空洞卷积相关的填充设置，如果dilation大于1，则按照空洞卷积的计算规则设置合适的填充pad_dilconv，
    # 否则使用普通卷积的填充值 pad_conv。
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv
    # 定义了一个内部函数 conv_i，用于返回具有指定参数（卷积核数量、卷积核大小、扩张率等）的卷积层实例，
    # 方便后续循环构建多层卷积结构时重复使用。
    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)
    # 创建了第一个卷积层 conv_1 和最后一个卷积层 conv_n，中间卷积层的构建通过循环来实现。
    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)
    
    # 修改卷积层以支持复数输入
    # conv_1.bias = conv_1.bias.to(torch.complex64)
    # conv_n.bias = conv_n.bias.to(torch.complex64)
    # conv_i().bias = conv_i().bias.to(torch.complex64)
    # print("conv_1.bias type:", conv_1.bias.type())
    # print("conv_n.bias type:", conv_n.bias.type())
    # print("conv_i().bias type:", conv_i().bias.type())


    # relu 根据传入的 nl 参数确定具体使用的激活函数（relu 或 lrelu），并将其赋值给 nll 变量。
    nll = relu if nl == 'relu' else lrelu

    '''
    构建整个卷积块的层列表layers，先添加第一个卷积层conv_1 和对应的激活函数实例，然后通过循环添加中间的卷积层（若bn为True，
    还会添加批归一化层）以及激活函数实例，最后添加最后一个卷积层conv_n，最终将这个层列表包装成nn.Sequential 顺序容器并返回，
    形成一个完整的卷积块模块，可以方便地作为神经网络中的一个基本构建单元使用。
    '''
    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class DnCn(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        # 调用父类（nn.Module）的初始化方法，确保正确初始化模块相关的属性。
        super(DnCn, self).__init__()
        # 保存传入的卷积块数量 nc 和每个卷积块中卷积层数 nd 的参数值，方便在后续的前向传播等方法中使用。
        self.nc = nc
        self.nd = nd
        # 打印创建的模块名称相关信息，格式为 D[nd]C[nc]，用于在初始化时输出提示信息，便于查看模型构建情况。
        print('Creating D{}C{}'.format(nd, nc))
        # 初始化两个空列表 conv_blocks 和 dcs，分别用于存储卷积块模块和数据一致性（Data Consistency）模块实例。
        conv_blocks = []
        dcs = []
        # 将前面定义的 conv_block 函数赋值给 conv_layer 变量，这样后续可以通过调用 conv_layer 来方便地创建卷积块实例。
        conv_layer = conv_block
        '''
        通过循环 for i in range(nc)：
        每次循环调用 conv_layer 并传入相应参数（输入通道数n_channels、卷积层数 nd 以及其他关键字参数 **kwargs）
        创建一个卷积块实例，添加到 conv_blocks 列表中。
        同时创建一个cl.DataConsistencyInKspace（假设cl是一个相关的模块库，该类用于处理数据一致性相关操作，
        norm='ortho' 表示使用正交归一化方式）实例，并添加到 dcs 列表中。
        '''
        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))
        '''
        最后将conv_blocks和dcs列表分别包装成nn.ModuleList类型赋值给self.conv_blocks 和 self.dcs属性，
        nn.ModuleList 是 PyTorch 中用于存储多个模块实例的容器，方便管理和操作多个模块对象。
        '''
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        # 通过 for i in range(self.nc) 循环进行 nc 次迭代，对应每个卷积块和数据一致性模块的操作
        for i in range(self.nc):
            # 取出第 i 个卷积块对输入 x 进行卷积操作，得到卷积后的结果 x_cnn。
            x_cnn = self.conv_blocks[i](x)
            # 将原始输入 x 和卷积结果 x_cnn 相加，实现残差连接（Residual Connection），有助于梯度传播和训练深度网络。
            x = x + x_cnn
            # 调用第 i 个数据一致性模块的 perform 方法，传入当前的特征 x、k（可能是 k - 空间相关数据）和
            # m（可能是掩码等相关数据）进行数据一致性处理，更新 x 的值。
            x = self.dcs[i].perform(x, k, m)

        return x


class StochasticDnCn(DnCn):
    def __init__(self, n_channels=2, nc=5, nd=5, p=None, **kwargs):
        #调用父类DnCn的初始化方法，继承父类的属性和方法，同时传入相应参数进行初始化，确保和父类一样构建好卷积块和数据一致性模块等基础结构。
        super(StochasticDnCn, self).__init__(n_channels, nc, nd, **kwargs)
        #初始化一个布尔变量self.sample，用于控制是否进行随机采样（丢弃连接）的行为，初始值设为False，表示默认不进行这种随机操作。
        self.sample = False
        '''
        接收一个概率参数 p，用于控制每个卷积块连接随机丢弃的概率。如果 p 为 None，则通过self.p = np.linspace(0, 0.5, nc) 
        创建一个从 0 到 0.5 均匀分布的 nc 个概率值的数组，用于为每个卷积块指定不同的随机丢弃连接概率，
        其中 nc 是从父类继承来的卷积块数量参数，最后打印出这个概率数组 self.p，方便查看具体的概率设置情况。
        '''
        self.p = p
        if not p:
            self.p = np.linspace(0, 0.5, nc)
        print(self.p)

    def forward(self, x, k, m):
        for i in range(self.nc):

            # stochastically drop connection
            if self.training or self.sample:
                if np.random.random() <= self.p[i]:
                    continue

            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x

    def set_sample(self, flag=True):
        self.sample = flag


class DnCn3D(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn3D, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{} (3D)'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


class DnCn3DDS(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, fr_d=None, clipped=False, mode='pytorch', **kwargs):
        """

        Parameters
        ----------

        fr_d: frame distance for data sharing layer. e.g. [1, 3, 5]

        """
        super(DnCn3DDS, self).__init__()
        self.nc = nc
        self.nd = nd
        self.mode = mode
        print('Creating D{}C{}-DS (3D)'.format(nd, nc))
        if self.mode == 'theano':
            print('Initialised with theano mode (backward-compatibility)')
        conv_blocks = []
        dcs = []
        kavgs = []

        if not fr_d:
            fr_d = list(range(10))
        self.fr_d = fr_d

        conv_layer = conv_block

        # update input-output channels for data sharing
        n_channels = 2 * len(fr_d)
        n_out = 2
        kwargs.update({'n_out': 2})

        for i in range(nc):
            kavgs.append(cl.AveragingInKspace(fr_d, i>0, clipped, norm='ortho'))
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)
        self.kavgs = nn.ModuleList(kavgs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_ds = self.kavgs[i](x, m)
            if self.mode == 'theano':
                # transpose the layes
                x_ds_tmp = torch.zeros_like(x_ds)
                nneigh = len(self.fr_d)
                for j in range(nneigh):
                    x_ds_tmp[:,2*j] = x_ds[:,j]
                    x_ds_tmp[:,2*j+1] = x_ds[:,j+nneigh]
                x_ds = x_ds_tmp

            x_cnn = self.conv_blocks[i](x_ds)
            x = x + x_cnn
            x = self.dcs[i](x, k, m)

        return x


class DnCn3DShared(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn3DShared, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{}-S (3D)'.format(nd, nc))

        self.conv_block = conv_block(n_channels, nd, **kwargs)
        self.dc = cl.DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_block(x)
            x = x + x_cnn
            x = self.dc.perform(x, k, m)

        return x


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        # self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # # add iteration hidden connection
        # self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # self.i2h.weight = torch.nn.Parameter(self.i2h.weight.to(torch.complex64))
        # self.i2h.bias = torch.nn.Parameter(self.i2h.bias.to(torch.complex64))
        self.i2h.weight = torch.nn.Parameter(self.i2h.weight.to(torch.float32))
        self.i2h.bias = torch.nn.Parameter(self.i2h.bias.to(torch.float32))

        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # self.h2h.weight = torch.nn.Parameter(self.h2h.weight.to(torch.complex64))
        # self.h2h.bias = torch.nn.Parameter(self.h2h.bias.to(torch.complex64))

        self.h2h.weight = torch.nn.Parameter(self.h2h.weight.to(torch.float32))
        self.h2h.bias = torch.nn.Parameter(self.h2h.bias.to(torch.float32))

        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # self.ih2ih.bias = torch.nn.Parameter(self.ih2ih.bias.to(torch.complex64))
        # self.ih2ih.weight = torch.nn.Parameter(self.ih2ih.weight.to(torch.complex64))

        self.ih2ih.weight = torch.nn.Parameter(self.ih2ih.weight.to(torch.float32))
        self.ih2ih.bias = torch.nn.Parameter(self.ih2ih.bias.to(torch.float32))

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        # CRNNcell-input: torch.Size([4, 2, 256, 32]) 4是batch_size
        # print('CRNNcell-input:',input.shape)
        # CRNNcell-input: torch.complex64
        
        # CRNNcell-input: torch.float32
        # print('CRNNcell-input:',input.dtype)
        in_to_hid = self.i2h(input)
        # CRNNcell-hidden: torch.float32
        # print('CRNNcell-hidden:',hidden.dtype)
        hid_to_hid = self.h2h(hidden)
        # print('CRNNcell-hidden_iteration:',hidden_iteration.dtype)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False):
        # BCRNNlayer-forward-input: torch.complex64
        # print('BCRNNlayer-forward-input:',input.dtype)
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            # print('BCRNNlayer-forward-input[i]:',input[i].dtype)
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNN_MRI(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    # def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
    def __init__(self, config):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI, self).__init__()
        config = config.CRNN_MRI
        self.n_ch = config.n_ch
        self.nc = config.nc
        self.nd = config.nd
        self.nf = config.nf
        self.ks = config.ks

        self.bcrnn = BCRNNlayer(self.n_ch, self.nf, self.ks)
        # self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        # self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        # self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        # self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        # self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        # self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        # self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        
        # 所有卷积层的权重和偏置都转换为了复数类型。请确保在模型初始化时执行这些转换。
        self.conv1_x = nn.Conv2d(self.nf, self.nf, self.ks, padding=self.ks//2)
        # self.conv1_x.weight = self.conv1_x.weight.to(torch.complex64)
        # self.conv1_x.bias = self.conv1_x.bias.to(torch.complex64)
        # self.conv1_x.bias = torch.nn.Parameter(self.conv1_x.bias.to(torch.complex64))
        self.conv1_x.weight = torch.nn.Parameter(self.conv1_x.weight.to(torch.float32))
        self.conv1_x.bias = torch.nn.Parameter(self.conv1_x.bias.to(torch.float32))

        self.conv1_h = nn.Conv2d(self.nf, self.nf, self.ks, padding=self.ks//2)
        # self.conv1_h.weight = self.conv1_h.weight.to(torch.complex64)
        # self.conv1_h.bias = self.conv1_h.bias.to(torch.complex64)
        # self.conv1_h.bias = torch.nn.Parameter(self.conv1_h.bias.to(torch.complex64))
        self.conv1_h.weight = torch.nn.Parameter(self.conv1_h.weight.to(torch.float32))
        self.conv1_h.bias = torch.nn.Parameter(self.conv1_h.bias.to(torch.float32))

        self.conv2_x = nn.Conv2d(self.nf, self.nf, self.ks, padding=self.ks//2)
        # self.conv2_x.weight = self.conv2_x.weight.to(torch.complex64)
        # self.conv2_x.bias = self.conv2_x.bias.to(torch.complex64)
        # self.conv2_x.bias = torch.nn.Parameter(self.conv2_x.bias.to(torch.complex64))

        self.conv2_h = nn.Conv2d(self.nf, self.nf, self.ks, padding=self.ks//2)
        # self.conv2_h.weight = self.conv2_h.weight.to(torch.complex64)
        # self.conv2_h.bias = self.conv2_h.bias.to(torch.complex64)
        # self.conv2_h.bias = torch.nn.Parameter(self.conv2_h.bias.to(torch.complex64))
        self.conv2_h.weight = torch.nn.Parameter(self.conv2_h.weight.to(torch.float32))
        self.conv2_h.bias = torch.nn.Parameter(self.conv2_h.bias.to(torch.float32))

        self.conv3_x = nn.Conv2d(self.nf, self.nf, self.ks, padding=self.ks//2)
        # self.conv3_x.weight = self.conv3_x.weight.to(torch.complex64)
        # self.conv3_x.bias = self.conv3_x.bias.to(torch.complex64)
        # self.conv3_x.bias = torch.nn.Parameter(self.conv3_x.bias.to(torch.complex64))
        self.conv3_x.weight = torch.nn.Parameter(self.conv3_x.weight.to(torch.float32))
        self.conv3_x.bias = torch.nn.Parameter(self.conv3_x.bias.to(torch.float32))

        self.conv3_h = nn.Conv2d(self.nf, self.nf, self.ks, padding=self.ks//2)
        # self.conv3_h.weight = self.conv3_h.weight.to(torch.complex64)
        # self.conv3_h.bias = self.conv3_h.bias.to(torch.complex64)
        # self.conv3_h.bias = torch.nn.Parameter(self.conv3_h.bias.to(torch.complex64))
        self.conv3_h.weight = torch.nn.Parameter(self.conv3_h.weight.to(torch.float32))
        self.conv3_h.bias = torch.nn.Parameter(self.conv3_h.bias.to(torch.float32))

        self.conv4_x = nn.Conv2d(self.nf, self.n_ch, self.ks, padding=self.ks//2)
        # self.conv4_x.weight = self.conv4_x.weight.to(torch.complex64)
        # self.conv4_x.bias = self.conv4_x.bias.to(torch.complex64)
        # self.conv4_x.bias = torch.nn.Parameter(self.conv4_x.bias.to(torch.complex64))
        self.conv4_x.weight = torch.nn.Parameter(self.conv4_x.weight.to(torch.float32))
        self.conv4_x.bias = torch.nn.Parameter(self.conv4_x.bias.to(torch.float32))

        
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(self.nc):
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, test=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        '''
        CRNN_MRI-forward-x: torch.float32
        CRNN_MRI-forward-x: torch.Size([4, 2, 256, 32, 30])
        CRNN_MRI-forward-k: torch.Size([4, 2, 256, 32, 30])
        CRNN_MRI-forward-m: torch.Size([4, 2, 256, 32, 30])
        '''
        # print('CRNN_MRI-forward-x:',x.dtype)  # 检查输入数据类型
        # print('CRNN_MRI-forward-x:',x.shape)  # 检查输入数据类型
        # print('CRNN_MRI-forward-k:',k.shape)  # 检查输入数据类型
        # print('CRNN_MRI-forward-m:',m.shape)  # 检查输入数据类型
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        for i in range(1,self.nc+1):

            x = x.permute(4,0,1,2,3)
            x = x.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch,self.nf,width, height)
            net['t%d_x0'%i] = self.bcrnn(x, net['t%d_x0'%(i-1)], test)
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1,self.nf,width, height)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            # CRNN_MRI-x-1:dtype torch.float32
            # CRNN_MRI-x-1:shape torch.Size([120, 2, 256, 32])
            # print('CRNN_MRI-x-1:dtype',x.dtype)
            # print('CRNN_MRI-x-1:shape',x.shape)
            net['t%d_out'%i] = x + net['t%d_x4'%i]

            net['t%d_out'%i] = net['t%d_out'%i].view(-1,n_batch, n_ch, width, height)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,3,4,0)
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = self.dcs[i-1].perform(net['t%d_out'%i], k, m)
            x = net['t%d_out'%i]
            # CRNN_MRI-x-2:dtype torch.complex64
            # CRNN_MRI-x-2:shape torch.Size([4, 2, 256, 32, 30])
            # print('CRNN_MRI-x-2:dtype',x.dtype)
            # print('CRNN_MRI-x-2:shape',x.shape)

            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net['t%d_out'%i]


