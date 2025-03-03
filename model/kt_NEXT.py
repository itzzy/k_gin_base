import torch
import torch.nn as nn
# from network.layers import *
from model.kt_layers import *

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def xf_CNN(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):
    """
    xf-CNN block in x-f domain
    """

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=dilation, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class CRNN_MRI(nn.Module):
    """
    CRNN-MRI block in image domain
    RNN evolves over temporal dimension only
    """
    def __init__(self, n_ch, nf=64, ks=3, dilation=2):
        super(CRNN_MRI, self).__init__()
        self.nf = nf
        self.ks = ks

        self.bcrnn_1 = BCRNNlayer(n_ch, nf, ks, dilation=1)
        self.bcrnn_2 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_3 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_4 = BCRNNlayer(nf, nf, ks, dilation)

        self.conv4_x = nn.Conv2d(nf, 2, ks, padding=ks//2)

    def forward(self, x, test=False):

        n_batch, n_ch, width, length, n_seq = x.size()

        x = x.permute(4, 0, 1, 2, 3)

        out = self.bcrnn_1(x, None, test)
        # print('CRNN_MRI-out-shape-1:',out.shape) #torch.Size([18, 4, 64, 192, 192])
        out = self.bcrnn_2(out, None, test)
        out = self.bcrnn_3(out, None, test)
        out = self.bcrnn_4(out, None, test)
        out = out.view(-1, self.nf, width, length)
        out = self.conv4_x(out)

        out = out.view(-1, n_batch, 2, width, length)
        out = out.permute(1, 2, 3, 4, 0)

        return out


class kt_NEXT_model(nn.Module):
    """
    network architecture for k-t NEXT
    """
    # def __init__(self, n_channels=2, nd=5, nc=5, nf=64, dilation=3):
    def __init__(self, config):
        super(kt_NEXT_model, self).__init__()
        # self.nc = nc
        # self.dilation = dilation
        config = config.kt_NEXT_model
        self.n_channels = config.n_channels
        self.nd = config.nd
        self.nc = config.nc
        self.nf = config.nf
        self.dilation = config.dilation
        xf_conv_blocks = []
        xt_conv_blocks = []
        dcs_xf = []
        dcs_xt = []
        tdxf = []

        for i in range(self.nc):
            # xf_conv_blocks.append(xf_CNN(2, nd, nf, n_out=n_channels, conv_dim=2, dilation=self.dilation))
            # xt_conv_blocks.append(CRNN_MRI(n_channels, nf, dilation=self.dilation))
            xf_conv_blocks.append(xf_CNN(2, self.nd, self.nf, n_out=self.n_channels, conv_dim=2, dilation=self.dilation))
            xt_conv_blocks.append(CRNN_MRI(self.n_channels, self.nf, dilation=self.dilation))
            dcs_xf.append(DataConsistencyInKspace(norm='ortho'))
            dcs_xt.append(DataConsistencyInKspace(norm='ortho'))
            tdxf.append(TransformDataInXfSpaceTA(i > 0, norm=True))

        self.xf_conv_blocks = nn.ModuleList(xf_conv_blocks)
        self.xt_conv_blocks = nn.ModuleList(xt_conv_blocks)
        self.dcs_xf = nn.ModuleList(dcs_xf)
        self.dcs_xt = nn.ModuleList(dcs_xt)
        self.tdxf = nn.ModuleList(tdxf)

    def forward(self, x, k, m):
        # print('kt_model-x-shape:',x.shape) #torch.Size([4, 2, 192, 192, 18])
        # print('kt_model-k-shape:',k.shape) #torch.Size([4, 2, 192, 192, 18])
        # print('kt_model-m-shape:',m.shape) #torch.Size([4, 2, 192, 192, 18])
        net = {}
        net_xf = {}
        self.normalized = True
        for i in range(self.nc):
            # x-f domain reconstruction
            xf, xf_avg = self.tdxf[i].perform(x, k, m)
            nb, nc, nx, ny, nt = xf.shape
            # xf = xf.permute(0, 3, 1, 2, 4)
            xf = xf.permute(0, 3, 1, 2, 4).contiguous()
            xf = xf.view(-1, nc, nx, nt)
            xf_out = self.xf_conv_blocks[i](xf)
            xf_out = xf_out.view(-1, ny, 2, nx, nt)
            xf_out = xf_out.permute(0, 2, 3, 1, 4)  # (n, nc, nx, ny, nt)
            xf_out = xf_out + xf_avg
            # print('kt_NEXT_model-forward-xf_out-shape:',xf_out.shape) #torch.Size([1, 2, 192, 192, 18])
            # print('kt_NEXT_model-forward-xf_out-dtype:',xf_out.dtype) #torch.float32
            xf_out = xf_out.permute(0,2,3,4,1)
            xf_out_complex = torch.view_as_complex(xf_out.contiguous())  # 转换为复数
            # print('kt_NEXT_model-forward-xf_out_complex-shape:',xf_out_complex.shape) # torch.Size([1, 256, 256, 30])
            # print('kt_NEXT_model-forward-xf_out_complex-dtype:',xf_out_complex.dtype) #torch.complex64
            # transform signal from x-f domain to image domain
            # out_img = fftshift_pytorch(torch.ifft(ifftshift_pytorch(xf_out.permute(0, 2, 3, 4, 1), axes=[-2]), 1, normalized=True), axes=[-2])
            # out_img = fftshift_pytorch(
            #             torch.fft.ifft(
            #                 ifftshift_pytorch(xf_out.permute(0, 2, 3, 4, 1), axes=[-2]), 
            #                 dim=-2, 
            #                 norm='ortho'  # 对应 normalized=True
            #             ), 
            #             axes=[-2]
            #         )
            
            # x_diff_complex = torch.view_as_complex(x_diff.contiguous())  # 转换为复数
            xf_out_fft = torch.fft.fft(
                        ifftshift_pytorch(xf_out_complex, axes=[-2]),
                        dim=-2,
                        norm='ortho' if self.normalized else None
                        )
            out_img = fftshift_pytorch(xf_out_fft, axes=[-2])
            out_img = torch.view_as_real(out_img)  # 拆分为实部和虚部 [n, nx, ny, nt, 2]
            out_img = out_img.permute(0, 4, 1, 2, 3)
            
            # print('kt_NEXT_model-forward-out_img-shape:',out_img.shape) #torch.Size([1, 2, 256, 256, 30])
            # print('kt_NEXT_model-forward-out_img-dtype:',out_img.dtype) #torch.float32
            # out_img = torch.view_as_real(out_img)  # 强制转换为实数张量 [..., 2]
            # out_img = out_img.permute(0, 4, 1, 2, 3)
            x = self.dcs_xf[i].perform(out_img, k, m)

            # image domain reconstruction
            out = self.xt_conv_blocks[i](x)
            x = x + out
            x = self.dcs_xt[i].perform(x, k, m)
            net['t%d' % i] = x
            net_xf['t%d' % i] = xf_out
            
            # print("img-shape:",img['t%d' % (nc - 1)].shape) #torch.Size([1, 2, 256, 256, 30])
            # print("xf_out-shape:",xf_out['t%d' % (nc-1)].shape) #torch.Size([1, 256, 256, 30, 2])

        return net_xf, net
