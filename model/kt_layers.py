import torch
import torch.nn as nn
from torch.autograd import Variable


def _fftshift(x, axes, offset=1):
    """ Apply ifftshift to x.

    Parameters:
    -----------

    x: torch.Tensor

    axes: tuple. axes to apply ifftshift. E.g.: axes=(-1), axes=(2,3), etc..

    Returns:
    --------

    result of applying ifftshift(x, axes).

    """
    # build slice
    x_shape = x.shape
    ndim = len(x_shape)
    axes = [ (ndim + ax) % ndim for ax in axes ]

    # apply shift for each axes:
    for ax in axes:
        # build slice:
        if x_shape[ax] == 1:
            continue
        n = x_shape[ax]
        half_n = (n + offset)//2
        curr_slice = [ slice(0, half_n) if i == ax else slice(x_shape[i]) for i in range(ndim) ]
        curr_slice_2 = [ slice(half_n, x_shape[i]) if i == ax else slice(x_shape[i]) for i in range(ndim) ]
        x = torch.cat([x[curr_slice_2], x[curr_slice]], dim=ax)
    return x


def fftshift_pytorch(x, axes):
    return _fftshift(x, axes, offset=1)


def ifftshift_pytorch(x, axes):
    return _fftshift(x, axes, offset=0)


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    # 在计算前检查输入
    assert not torch.isnan(k).any(), "NaN in k"
    assert not torch.isnan(k0).any(), "NaN in k0"
    assert not torch.isnan(mask).any(), "NaN in mask"
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    # 检查中间结果
    if torch.isnan(out).any():
        print("NaN in data_consistency out")
    return out

def complex_multiply(x, y, u, v):
    """
    Computes (x+iy) * (u+iv) = (x * u - y * v) + (x * v + y * u)i = z1 + iz2

    Returns (real z1, imaginary z2)
    """

    z1 = x * u - y * v
    z2 = x * v + y * u

    return torch.stack((z1, z2), dim=-1)


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        # k = torch.fft(x, 2, normalized=self.normalized)
        # out = data_consistency(k, k0, mask, self.noise_lvl)
        # x_res = torch.ifft(out, 2, normalized=self.normalized)
        # print('DataConsistencyInKspace-perform-x-shape:',x.shape) #torch.Size([4, 18, 192, 192, 2])
        # print('DataConsistencyInKspace-perform-x-dtype:',x.dtype) # torch.float32
        # print('DataConsistencyInKspace-perform-k0-shape:',k0.shape) #torch.Size([4, 18, 192, 192, 2])
        # print('DataConsistencyInKspace-perform-k0-dtype:',k0.dtype) # torch.float32
        # print('DataConsistencyInKspace-perform-mask-shape:',mask.shape) #torch.Size([4, 18, 192, 192, 2])
        # print('DataConsistencyInKspace-perform-mask-dtype:',mask.dtype) #torch.float32
        # 将输入的实数数据转换为复数（假设x的最后一个维度是2，对应实部和虚部） 
        x_complex = torch.view_as_complex(x.contiguous())

        # 执行2D FFT，使用dim指定空间维度（假设nx, ny是最后两个维度）
        k = torch.fft.fft2(x_complex, dim=(-2, -1), norm='ortho' if self.normalized else None)
        k0_complex = torch.view_as_complex(k0.contiguous())
        mask_complex = torch.view_as_complex(x.contiguous())
        # 数据一致性操作（需确保data_consistency函数支持复数）
        # out = data_consistency(k, k0, mask, self.noise_lvl)
        ###都转为复数，可能不对
        out = data_consistency(k, k0_complex, mask_complex, self.noise_lvl)
        # print('DataConsistencyInKspace-perform-out-shape:',out.shape) #torch.Size([4, 18, 192, 192])
        # print('DataConsistencyInKspace-perform-out-dtype:',out.dtype) #torch.complex64
        # 执行2D逆FFT
        x_res_complex = torch.fft.ifft2(out, dim=(-2, -1), norm='ortho' if self.normalized else None)

        # 将复数结果拆分为实部和虚部
        x_res = torch.view_as_real(x_res_complex)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    iteration: True or False, to use iteration recurrence or not; if iteration=False: hidden_iteration=None

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # add iteration hidden connection
        if self.iteration:
            self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden, hidden_iteration=None):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)
            hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid + hid_to_hid)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
               iteration: True if use iteration recurrence and input_iteration is not None; False if input_iteration=None

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.iteration = iteration
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, dilation, iteration=self.iteration)

    def forward(self, input, input_iteration=None, test=False):
        # print('BCRNN-input-shape:',input.shape) #torch.Size([18, 4, 2, 192, 192])
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h), requires_grad=False).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h), requires_grad=False).cuda()

        output_f = []
        output_b = []
        if input_iteration is not None:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden, input_iteration[i])
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden, input_iteration[nt - i -1])
                output_b.append(hidden)
        else:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden)
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden)
                output_b.append(hidden)
        '''
        报错:
        File "/nfs/zzy/code/kt-Dynamic-MRI/network/kt_NEXT.py", line 77, in forward
        out = self.bcrnn_2(out, None, test)
        File "/root/anaconda3/envs/myvllm/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
        File "/root/anaconda3/envs/myvllm/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
        File "/nfs/zzy/code/kt-Dynamic-MRI/network/layers.py", line 211, in forward
        nt, nb, nc, nx, ny = input.shape
        ValueError: not enough values to unpack (expected 5, got 4)
        '''
        # output_f = torch.cat(output_f)
        # output_b = torch.cat(output_b[::-1])
        # 修改此处：将torch.cat替换为torch.stack
        output_f = torch.stack(output_f, dim=0)  # 形状变为 (nt, nb, hidden_size, nx, ny)
        output_b = torch.stack(output_b[::-1], dim=0)

        output = output_f + output_b
        # print('BCRNNlayer-output-shape-1:',output.shape) #torch.Size([18, 4, 64, 192, 192])
        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)
        # print('BCRNNlayer-output-shape-2:',output.shape) #torch.Size([18, 4, 64, 192, 192])
        return output

class CRNN_i(nn.Module):
    """
    Convolutional RNN cell that evolves over iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(CRNN_i, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # add iteration hidden connection
        if iteration:
            self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden_iteration=None):
        in_to_hid = self.i2h(input)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)
            hidden = self.relu(in_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid)

        return hidden

class TransformDataInXfSpaceTA(nn.Module):

    def __init__(self, divide_by_n=False, norm=True):
        super(TransformDataInXfSpaceTA, self).__init__()
        self.normalized = norm
        self.divide_by_n = divide_by_n

    def forward(self, x, k0, mask):
        return self.perform(x, k0, mask)

    def perform(self, x, k0, mask):
        """
        transform to x-f space with subtraction of average temporal frame
        :param x: input image with shape [n, 2, nx, ny, nt]
        :param mask: undersampling mask
        :return: difference data; DC baseline
        """
        # temporally average kspace and image data
        x = x.permute(0, 4, 2, 3, 1)
        mask = mask.permute(0, 4, 2, 3, 1)
        k0 = k0.permute(0, 4, 2, 3, 1)
        x_complex = torch.view_as_complex(x.contiguous())
        k0_complex = torch.view_as_complex(k0.contiguous())
        mask_complex = torch.view_as_complex(mask.contiguous())
        
        # k = torch.fft(x, 2, normalized=self.normalized)
        dim=(-2, -1)
        k_x_complex = torch.fft.fft2(x_complex, dim=dim, norm='ortho')
        # print('perform-k_x_comlex-shape:',k_x_complex.shape) #torch.Size([4, 18, 192, 192])
        # print('perform-k_x_comlex-dtype:',k_x_complex.dtype) # torch.complex64
        k = torch.view_as_real(k_x_complex)
        if self.divide_by_n:
            # print('self.divide_by_n')
            '''
            self.divide_by_n 打印如下:
            perform-k_avg-shape-2: torch.Size([4, 192, 192])
            perform-k_avg-dtype:-2 torch.complex64
            perform-k_avg-shape-3: torch.Size([4, 18, 192, 192, 2])
            perform-k_avg-dtype:-3 torch.float32
            '''
            # k_avg = torch.div(torch.sum(k, 1), k.shape[1])
            k_avg = torch.div(torch.sum(k_x_complex, 1), k_x_complex.shape[1])
        else:
            # print('else----------')
            '''
            else时打印如下:
            else----------
            perform-k_avg-shape-1: torch.Size([4, 192, 192, 2])
            perform-k_avg-dtype-1: torch.float32
            perform-k_avg-shape-2: torch.Size([4, 192, 192, 2])
            perform-k_avg-dtype:-2 torch.float32
            perform-k_avg-shape-3: torch.Size([4, 18, 192, 192, 2])
            perform-k_avg-dtype:-3 torch.float32
            '''
            k_avg = torch.div(torch.sum(k, 1), torch.clamp(torch.sum(mask, 1), min=1))
            # print('perform-k_avg-shape-1:',k_avg.shape) #torch.Size([4, 192, 192, 2])
            # print('perform-k_avg-dtype-1:',k_avg.dtype) #torch.float32
            # k_avg = torch.div(torch.sum(k_x_complex, 1), torch.clamp(torch.sum(mask_complex, 1), min=1))
        # print('perform-k_avg-shape-2:',k_avg.shape) #torch.Size([4, 192, 192])
        # print('perform-k_avg-dtype:-2',k_avg.dtype) #torch.complex64
        # k_avg = torch.view_as_real(k_avg)
        # 确保 k_avg 是实虚部分离的浮点张量
        if k_avg.dtype in [torch.complex64, torch.complex128]:
            k_avg = torch.view_as_real(k_avg)  # [..., 2]
        nb, nx, ny, nc = k_avg.shape
        k_avg = k_avg.view(nb, 1, nx, ny, nc)
        # repeat the temporal frame and
        # k_avg = k_avg.repeat(1, k.shape[1], 1, 1, 1)
        k_avg = k_avg.repeat(1, k_x_complex.shape[1], 1, 1, 1)
        # print('perform-k_avg-shape-3:',k_avg.shape) #torch.Size([4, 18, 192, 192, 2])
        # print('perform-k_avg-dtype:-3',k_avg.dtype) #torch.float32
        k_avg_complex = torch.view_as_complex(k_avg.contiguous())
        
        # subtract the temporal average frame
        k_diff = torch.sub(k, k_avg)
        k_diff = torch.view_as_complex(k_diff.contiguous())
        # k_diff = torch.sub(k_x_complex, k_avg)
        # x_diff = torch.ifft(k_diff, 2, normalized=self.normalized)
        x_diff = torch.fft.ifft2(k_diff, dim=(-2, -1), norm='ortho')
        x_diff = torch.view_as_real(x_diff)
        # transform to x-f space to get the baseline
        # k_avg = data_consistency(k_avg, k0, mask)
        # k_avg = data_consistency(k_avg, k0_complex, mask_complex)
        k_avg = data_consistency(k_avg_complex, k0_complex, mask_complex)
        # x_avg = torch.ifft(k_avg, 2, normalized=self.normalized)
        x_avg = torch.fft.ifft2(k_avg, dim=(-2, -1), norm='ortho')
        x_avg = torch.view_as_real(x_avg)
        x_avg = x_avg.permute(0, 2, 3, 1, 4)  # [n, nx, ny, nt, 2]
        # x_f_avg = fftshift_pytorch(torch.fft(ifftshift_pytorch(x_avg, axes=[-2]), 1, normalized=self.normalized), axes=[-2])
        # x_f_avg = fftshift_pytorch(
        #             torch.fft.fft(
        #                 ifftshift_pytorch(x_avg, axes=[-2]), 
        #                 dim=-2, 
        #                 norm='ortho' if self.normalized else None
        #             ), 
        #             axes=[-2]
        #         )
        x_avg_complex = torch.view_as_complex(x_avg.contiguous())  # 转换为复数
        x_avg_fft = torch.fft.fft(
                    ifftshift_pytorch(x_avg_complex, axes=[-2]),
                    dim=-2,
                    norm='ortho' if self.normalized else None
                    )
        x_f_avg = fftshift_pytorch(x_avg_fft, axes=[-2])
        x_f_avg = torch.view_as_real(x_f_avg)  # 拆分为实部和虚部 [n, nx, ny, nt, 2]
        x_f_avg = x_f_avg.permute(0, 4, 1, 2, 3)

        # difference data
        x_diff = x_diff.permute(0, 2, 3, 1, 4)  # [n, nx, ny, nt, 2]
        # x_f_diff = fftshift_pytorch(torch.fft(ifftshift_pytorch(x_diff, axes=[-2]), 1, normalized=self.normalized), axes=[-2])
        # x_f_diff = fftshift_pytorch(
        #             torch.fft.fft(
        #                 ifftshift_pytorch(x_diff, axes=[-2]), 
        #                 dim=-2, 
        #                 norm='ortho' if self.normalized else None
        #             ), 
        #             axes=[-2]
        #         )
        x_diff_complex = torch.view_as_complex(x_diff.contiguous())  # 转换为复数
        x_diff_fft = torch.fft.fft(
                    ifftshift_pytorch(x_diff_complex, axes=[-2]),
                    dim=-2,
                    norm='ortho' if self.normalized else None
                    )
        x_f_diff = fftshift_pytorch(x_diff_fft, axes=[-2])
        x_f_diff = torch.view_as_real(x_f_diff)  # 拆分为实部和虚部 [n, nx, ny, nt, 2]
        x_f_diff = x_f_diff.permute(0, 4, 1, 2, 3)
        # print('perform-x_f_diff-shape:',x_f_diff.shape) #torch.Size([1, 2, 256, 256, 30])
        # print('perform-x_f_diff-dtype:',x_f_diff.dtype) #torch.float32
        # print('perform-x_f_avg-shape:',x_f_avg.shape) #torch.Size([1, 2, 256, 256, 30])
        # print('perform-x_f_avg-dtype:',x_f_avg.dtype) #torch.float32

        return x_f_diff, x_f_avg

class TransformDataInXtSpaceTA_mc(nn.Module):

    def __init__(self, divide_by_n=False, norm=True):
        super(TransformDataInXtSpaceTA_mc, self).__init__()
        self.normalized = norm
        self.divide_by_n = divide_by_n

    def forward(self, x, k0, mask, sensitivity):
        return self.perform(x, k0, mask, sensitivity)

    def perform(self, x, k0, mask, sensitivity):
        """
        compute temporal averaged frames with data consistency in multi-coil setting with sensitivity maps
        :param x: input image with shape [nt, nx, ny, 2]
        :param mask: undersampling mask [nt, ns, nx, ny, 2]
        :param k0: undersampled k-space data [nt, ns, nx, ny, 2]
        :param sensitivity: sensitivity maps [nt, ns, nx, ny, 2]
        :return: temporal average frames
        """

        if self.divide_by_n:
            x = complex_multiply(x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1),
                                sensitivity[..., 0], sensitivity[..., 1])
            k = torch.fft(x, 2, normalized=self.normalized)
            k_avg = torch.div(torch.sum(k, 0), k.shape[0])
        else:
            k_avg = torch.div(torch.sum(k0, 0), torch.clamp(torch.sum(mask, 0), min=1))

        # data consistency for each frame
        k_avg = k_avg.expand(x.shape[0], -1, -1, -1, -1)
        k_avg = data_consistency(k_avg, k0, mask)

        x_avg = torch.ifft(k_avg, 2, normalized=True)

        Sx_avg = complex_multiply(x_avg[..., 0], x_avg[..., 1],
                                sensitivity[..., 0],
                                -sensitivity[..., 1]).sum(dim=1)
        return Sx_avg

class TransformDataInXfSpaceTA_mc(nn.Module):

    def __init__(self, divide_by_n=False, norm=True):
        super(TransformDataInXfSpaceTA_mc, self).__init__()
        self.normalized = norm
        self.divide_by_n = divide_by_n

    def perform(self, x, k0, mask, sensitivity):
        """
        transform to x-f space with subtraction of average temporal frame in multi-coil setting
        :param x: input image with shape [nt, nx, ny, 2]
        :param mask: undersampling mask [nt, ns, nx, ny, 2]
        :param k0: undersampled k-space data [nt, ns, nx, ny, 2]
        :param sensitivity: sensitivity maps [nt, ns, nx, ny, 2]
        :return: difference data; DC baseline
        """

        x = complex_multiply(x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1),
                            sensitivity[..., 0], sensitivity[..., 1])
        k = torch.fft(x, 2, normalized=self.normalized)
        if self.divide_by_n:
            k_avg = torch.div(torch.sum(k, 0), k.shape[0])
        else:
            k_avg = torch.div(torch.sum(k0, 0), torch.clamp(torch.sum(mask, 0), min=1))

        ns, nx, ny, nc = k_avg.shape
        k_avg = k_avg.view(1, ns, nx, ny, nc)
        k_avg = k_avg.repeat(k.shape[0], 1, 1, 1, 1)

        # subtract the temporal average frame
        k_diff = torch.sub(k, k_avg)
        x_diff = torch.ifft(k_diff, 2, normalized=self.normalized)
        Sx_diff = complex_multiply(x_diff[..., 0], x_diff[..., 1],
                                sensitivity[..., 0],
                                -sensitivity[..., 1]).sum(dim=1) # [nt, nx, ny, 2]

        # transform to x-f space to get the baseline
        x_avg = torch.ifft(k_avg, 2, normalized=self.normalized)
        Sx_avg = complex_multiply(x_avg[..., 0], x_avg[..., 1],
                                sensitivity[..., 0],
                                -sensitivity[..., 1]).sum(dim=1)

        Sx_avg = Sx_avg.permute(1, 2, 0, 3)  # [nx, ny, nt, 2]
        x_f_avg = fftshift_pytorch(torch.fft(ifftshift_pytorch(Sx_avg, axes=[-2]), 1, normalized=self.normalized), axes=[-2])
        x_f_avg = x_f_avg.permute(2, 0, 1, 3)

        # difference data
        Sx_diff = Sx_diff.permute(1, 2, 0, 3)  # [nx, ny, nt, 2]
        x_f_diff = fftshift_pytorch(torch.fft(ifftshift_pytorch(Sx_diff, axes=[-2]), 1, normalized=self.normalized), axes=[-2])
        x_f_diff = x_f_diff.permute(2, 0, 1, 3)

        return x_f_diff, x_f_avg
