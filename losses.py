import torch
import lpips


class CriterionBase(torch.nn.Module):
    def __init__(self, config):
        super(CriterionBase, self).__init__()
        self.loss_names = config.which
        self.loss_weights = config.loss_weights
        self.loss_list = []
        self.use_weighting_mask = config.use_weighting_mask
        self.cardiac_crop_quantitative_metric = config.cardiac_crop_quantitative_metric

        self.k_loss_list = config.k_recon_loss_combined.k_loss_list
        self.k_loss_weighting = [1] + [config.k_recon_loss_combined.k_loss_decay ** (len(self.k_loss_list) - i - 2) for i in range(len(self.k_loss_list)-1)]  # weighting of kmae should be one
        assert len(self.k_loss_weighting) == len(self.k_loss_list)
        self.k_loss_weighting = [i*j for i, j in zip(self.k_loss_weighting, config.k_recon_loss_combined.k_loss_weighting)]

        for loss_name in config.which:
            loss_args = eval(f'config.{loss_name}').__dict__
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def get_loss(self, loss_name, args_dict):
        if loss_name == 'photometric' or loss_name == 'k_recon_loss':
            return PhotometricLoss(**args_dict)
        elif loss_name == 'k_recon_loss_combined':
            k_recon_loss_list = []
            for k_loss in self.k_loss_list:
                if k_loss == 'L1':
                    k_recon_loss_list.append(torch.nn.L1Loss())
                elif k_loss == 'HDR':
                    k_recon_loss_list.append(HDRLoss(eps=args_dict['eps']))
                else:
                    raise NotImplementedError
            return k_recon_loss_list
        elif loss_name == 'HDR':
            return HDRLoss(**args_dict)
        elif loss_name == 'psnr':
            return PSNR(**args_dict)
        else:
            raise NotImplementedError


class HDRLoss(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, input, target, weights=None, reduce=True):
        if not input.is_complex():
            input = torch.view_as_complex(input)
        if not target.is_complex():
            target = torch.view_as_complex(target)
        error = input - target

        loss = (error.abs()/(input.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        return loss.mean()

class CriterionKGIN(CriterionBase, torch.nn.Module):
    def __init__(self, config):
        super(CriterionKGIN, self).__init__(config)
        self.only_maskout = config.only_maskout

    def forward(self, k_pred, k_ref, im_pred, im_ref, kspace_mask, mode='train'):
        loss_dict = {}
        if mode == 'train': assert len(k_pred) == len(self.k_loss_list)

        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            if loss_name == 'k_recon_loss_combined':
                loss_dict[loss_name] = 0
                for pred, k_loss_term, k_loss_weights in zip(k_pred, loss_term, self.k_loss_weighting):
                    k_loss = k_loss_term(pred, k_ref)
                    loss_dict[loss_name] += k_loss_weights * k_loss
            elif loss_name == 'k_recon_loss' or loss_name == 'HDR':
                loss = loss_term(k_pred[0], k_ref)
                loss_dict[loss_name] = loss_weight * loss
            else:
                loss = loss_term(im_pred, im_ref)
                loss_dict[loss_name] = loss_weight * loss
        return loss_dict
# class CriterionKGIN(CriterionBase, torch.nn.Module):
#     def __init__(self, config):
#         super(CriterionKGIN, self).__init__(config)
#         self.only_maskout = config.only_maskout

#     def forward(self, k_pred, k_ref, im_pred, im_ref, kspace_mask, mode='train'):
#         loss_dict = {}
#         # forward Length of k_pred-1: 1
#         # forward Length of k_loss_list-1: 4
#         # forward Length of k_pred-2: 4
#         # forward Length of k_loss_list-2: 4
#         print(f"forward Length of k_pred-1: {len(k_pred)}")
#         print(f"forward Length of k_loss_list-1: {len(self.k_loss_list)}")
        
#         # 动态调整 k_pred 的长度
#         if len(k_pred) < len(self.k_loss_list):
#             # print(f"Warning: k_pred length ({len(k_pred)}) is shorter than k_loss_list ({len(self.k_loss_list)}). Padding k_pred.")
#             padding = [k_pred[-1].clone().unsqueeze(0) for _ in range(len(self.k_loss_list) - len(k_pred))]
#             k_pred = torch.cat([k_pred] + padding, dim=0)

#         print(f"forward Length of k_pred-2: {len(k_pred)}")
#         print(f"forward Length of k_loss_list-2: {len(self.k_loss_list)}")
        
#         if mode == 'train': 
#             assert len(k_pred) == len(self.k_loss_list)
        
#         # 确保 im_ref 在 GPU 上
#         if im_ref.device != im_pred.device:
#             im_ref = im_ref.to(im_pred.device)
# #         Shape of im_pred: torch.Size([2, 192, 192, 18]), Shape of im_ref: torch.Size([2, 18, 192, 192])
# # Shape of im_pred_adjusted: torch.Size([2, 18, 192, 192]), Shape of im_ref_adjusted: torch.Size([2, 192, 192, 18])
#         # print(f"Shape of im_pred: {im_pred.shape}, Shape of im_ref: {im_ref.shape}")
#         # 调整 im_pred 的维度顺序，使其与 im_ref 一致
#         im_pred_adjusted = im_pred.permute(0, 3, 2, 1)  # 从 [batch_size, channels, height, width] 变为 [batch_size, height, width, channels]
#         # im_ref_adjusted = im_ref.permute(0, 2, 3, 1)  # 从 [batch_size, channels, height, width] 变为 [batch_size, height, width, channels]
#         im_ref_adjusted = im_ref # 从 [batch_size, channels, height, width] 变为 [batch_size, height, width, channels]
#         # print(f"Shape of im_pred_adjusted: {im_pred_adjusted.shape}, Shape of im_ref_adjusted: {im_ref_adjusted.shape}")

#         for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
#             if loss_name == 'k_recon_loss_combined':
#                 loss_dict[loss_name] = 0
#                 for pred, k_loss_term, k_loss_weights in zip(k_pred, loss_term, self.k_loss_weighting):
#                     # 调整 pred 的维度顺序，从 [height, width, channels] 变为 [batch_size, channels, height, width]
#                     pred = pred.unsqueeze(0)  # 添加 batch_size 维度
#                     pred = pred.permute(0, 3, 1, 2)  # 调整维度顺序
#                     # Shape of pred: torch.Size([1, 18, 192, 192]), Shape of k_ref: torch.Size([2, 18, 192, 192])
#                     # print(f"Shape of pred: {pred.shape}, Shape of k_ref: {k_ref.shape}")
#                     # 确保 pred 和 k_ref 的形状一致
#                     if pred.shape != k_ref.shape:
#                         pred = pred.expand_as(k_ref)  # 将 pred 的形状扩展为与 k_ref 相同
                    
#                     # 确保 pred 和 k_ref 位于同一设备上
#                     if pred.device != k_ref.device:
#                         pred = pred.to(k_ref.device)  # 将 pred 移动到 k_ref 的设备上
                  
#                     # 确保 pred 和 k_ref 启用了梯度
#                     pred = pred.requires_grad_(True)
#                     k_ref = k_ref.requires_grad_(True)
#                     print('pred.requires_grad-1:',pred.requires_grad)  # 检查 pred 是否启用了梯度
#                     print('k_ref.requires_grad-1:',k_ref.requires_grad)  # 检查 k_ref 是否启用了梯度
                    
#                     k_loss = k_loss_term(pred, k_ref)
#                     loss_dict[loss_name] += k_loss_weights * k_loss
#             elif loss_name == 'k_recon_loss' or loss_name == 'HDR':
#                 # 调整 pred 的维度顺序
#                 pred = k_pred[0].unsqueeze(0)  # 添加 batch_size 维度
#                 pred = pred.permute(0, 3, 1, 2)  # 调整维度顺序
#                 # 确保 pred 和 k_ref 的形状一致
#                 if pred.shape != k_ref.shape:
#                     pred = pred.expand_as(k_ref)  # 将 pred 的形状扩展为与 k_ref 相同
               
#                 # 确保 pred 和 k_ref 位于同一设备上
#                 if pred.device != k_ref.device:
#                     pred = pred.to(k_ref.device)  # 将 pred 移动到 k_ref 的设备上
               
#                 # 确保 pred 和 k_ref 启用了梯度
#                 pred = pred.requires_grad_(True)
#                 k_ref = k_ref.requires_grad_(True)
#                 print('pred.requires_grad-2:',pred.requires_grad)  # 检查 pred 是否启用了梯度
#                 print('k_ref.requires_grad-2:',k_ref.requires_grad)  # 检查 k_ref 是否启用了梯度
#                 loss = loss_term(pred, k_ref)
#                 loss_dict[loss_name] = loss_weight * loss
#             else:
#                 # 调整 im_pred 和 im_ref 的维度顺序
#                 # print(f"Shape of im_pred_adjusted: {im_pred_adjusted.shape}, Shape of im_ref_adjusted: {im_ref_adjusted.shape}")
#                 loss = loss_term(im_pred_adjusted, im_ref_adjusted)
#                 loss_dict[loss_name] = loss_weight * loss
#         return loss_dict

# class CriterionKGIN(CriterionBase, torch.nn.Module):
#     def __init__(self, config):
#         super(CriterionKGIN, self).__init__(config)
#         self.only_maskout = config.only_maskout

#     def forward(self, k_pred, k_ref, im_pred, im_ref, kspace_mask, mode='train'):
#         loss_dict = {}
#         print(f"forward Length of k_pred-1: {len(k_pred)}")
#         print(f"forward Length of k_loss_list-1: {len(self.k_loss_list)}")
#         # 动态调整 k_pred 的长度
#         if len(k_pred) < len(self.k_loss_list):
#             print(f"Warning: k_pred length ({len(k_pred)}) is shorter than k_loss_list ({len(self.k_loss_list)}). Padding k_pred.")
#             # padding = [k_pred[-1]] * (len(self.k_loss_list) - len(k_pred))  # 重复最后一个元素
#             # padding = torch.stack(padding)  # 将列表转换为张量
#             # k_pred = torch.cat((k_pred, padding), dim=0)  # 拼接到 k_pred
#             # padding = [k_pred[-1].clone()] * (len(self.k_loss_list) - len(k_pred))  # 重复最后一个元素
#             # k_pred = torch.cat(k_pred + padding, dim=0)  # 沿 batch 维度拼接
#             # 确保 padding 的维度和 k_pred 的每个元素一致
#             padding = [k_pred[-1].clone().unsqueeze(0) for _ in range(len(self.k_loss_list) - len(k_pred))]
#             k_pred = torch.cat([k_pred] + padding, dim=0)  # 使用列表组合后拼接


#         print(f"forward Length of k_pred-2: {len(k_pred)}")
#         print(f"forward Length of k_loss_list-2: {len(self.k_loss_list)}")
        
#         if mode == 'train': assert len(k_pred) == len(self.k_loss_list)

#         for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
#             if loss_name == 'k_recon_loss_combined':
#                 loss_dict[loss_name] = 0
#                 for pred, k_loss_term, k_loss_weights in zip(k_pred, loss_term, self.k_loss_weighting):
#                     # Shape of pred: torch.Size([2, 192, 192, 18]), Shape of k_ref: torch.Size([2, 18, 192, 192, 2])
#                     print(f"Shape of pred: {pred.shape}, Shape of k_ref: {k_ref.shape}")
#                     k_loss = k_loss_term(pred, k_ref)
#                     loss_dict[loss_name] += k_loss_weights * k_loss
#             elif loss_name == 'k_recon_loss' or loss_name == 'HDR':
#                 loss = loss_term(k_pred[0], k_ref)
#                 loss_dict[loss_name] = loss_weight * loss
#             else:
#                 loss = loss_term(im_pred, im_ref)
#                 loss_dict[loss_name] = loss_weight * loss
#         return loss_dict


class PhotometricLoss(torch.nn.Module):
    def __init__(self, mode):
        super(PhotometricLoss, self).__init__()
        assert mode in ('charbonnier', 'L1', 'L2', 'HDR')
        if mode == 'charbonnier':
            self.loss = CharbonnierLoss()
        elif mode == 'L1':
            self.loss = torch.nn.L1Loss(reduction='mean')
        elif mode == 'KspaceL1':
            raise NotImplementedError('KspaceL1 is not implemented yet')
        elif mode == 'L2':
            self.loss = CharbonnierLoss(eps=1.e-6, alpha=1)
        elif mode == 'HDR':
            self.loss = HDRLoss(eps=1.e-3)

    def forward(self, inputs, outputs):
        return self.loss(inputs, outputs)


class PSNR(torch.nn.Module):
    def __init__(self, max_value=1.0, magnitude_psnr=True):
        super(PSNR, self).__init__()
        self.max_value = max_value
        self.magnitude_psnr = magnitude_psnr

    def forward(self, u, g):
        """

        :param u: noised image
        :param g: ground-truth image
        :param max_value:
        :return:
        """
        if self.magnitude_psnr:
            u, g = torch.abs(u), torch.abs(g)
        batch_size = u.shape[0]
        diff = (u.reshape(batch_size, -1) - g.reshape(batch_size, -1))
        square = torch.conj(diff) * diff
        max_value = g.abs().max() if self.max_value == 'on_fly' else self.max_value
        if square.is_complex():
            square = square.real
        v = torch.mean(20 * torch.log10(max_value / torch.sqrt(torch.mean(square, -1))))
        return v


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, alpha=0.45):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, y):
        diff = x - y
        square = torch.conj(diff) * diff
        if square.is_complex():
            square = square.real
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.pow(square + self.eps, exponent=self.alpha))
        return loss
