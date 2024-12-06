import os
import sys
import pathlib
import torch
import glob
import tqdm
import time
from torch.utils.data import DataLoader
from dataset.dataloader import CINE2DT
from model.k_interpolator import KInterpolator
from losses import CriterionKGIN
from utils import count_parameters, Logger, adjust_learning_rate as adjust_lr, NativeScalerWithGradNormCount as NativeScaler, add_weight_decay
from utils import multicoil2single
import numpy as np

# import wandb
# import json

# from wandb.util import WandBHistoryJSONEncoder

# class MyEncoder(WandBHistoryJSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, complex):
#             return {"real": obj.real, "imag": obj.imag}
#         return super().default(obj)

# 设置环境变量 CUDA_VISIBLE_DEVICES
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用 GPU 1 和 GPU 4
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'  # 指定使用 GPU 7 和 GPU 3
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4'  # 指定使用 GPU 4 和 GPU 7
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 指定使用 GPU 4 和 GPU 6  
#0和3
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,5,7' 


# PyTorch建议在使用多线程时设置OMP_NUM_THREADS环境变量，以避免系统过载。
os.environ['OMP_NUM_THREADS'] = '1'
# 设置PYTORCH_CUDA_ALLOC_CONF环境变量，以减少CUDA内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #,0,1,2,4,5,6,7
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #,0,1,2,4,5,6,7
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Using device: {device}")
print(f"Actual GPU being used: torch.cuda.current_device() -> {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(device)}")

class TrainerAbstract:
    def __init__(self, config):
        super().__init__()
        self.config = config.general
        self.debug = config.general.debug
        # if self.debug: config.general.exp_name = 'test'
        if self.debug: config.general.exp_name = 'Test'
        self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.only_infer = config.general.only_infer
        self.num_epochs = config.training.num_epochs if config.general.only_infer is False else 1

        # data
        # train_ds = CINE2DT(config=config.data, mode='train')
        train_ds = CINE2DT(config=config.data, mode='val')
        test_ds = CINE2DT(config=config.data, mode='val')
        self.train_loader = DataLoader(dataset=train_ds, num_workers=config.training.num_workers, drop_last=False,
                                       pin_memory=True, batch_size=config.training.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_ds, num_workers=1, drop_last=False, batch_size=1, shuffle=False)

        # network
        self.network = getattr(sys.modules[__name__], config.network.which)(eval('config.network'))
        self.network.initialize_weights()
        self.network.cuda()
        print("Parameter Count: %d" % count_parameters(self.network))

        # optimizer
        param_groups = add_weight_decay(self.network, config.training.optim_weight_decay)
        self.optimizer = eval(f'torch.optim.{config.optimizer.which}')(param_groups, **eval(f'config.optimizer.{config.optimizer.which}').__dict__)

        if config.training.restore_ckpt: self.load_model(config.training)
        self.loss_scaler = NativeScaler()

    def load_model(self, args):

        if os.path.isdir(args.restore_ckpt):
            args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
        ckpt = torch.load(args.restore_ckpt)
        self.network.load_state_dict(ckpt['model'], strict=True)

        print("Resume checkpoint %s" % args.restore_ckpt)
        if args.restore_training:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            # self.loss_scaler.load_state_dict(ckpt['scaler'])
            print("With optim & sched!")

    def save_model(self, epoch):
        ckpt = {'epoch': epoch,
                'model': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scaler': self.loss_scaler.state_dict()
                }
        torch.save(ckpt, f'{self.experiment_dir}/model_{epoch+1:03d}.pth')


class TrainerKInterpolator(TrainerAbstract):

    def __init__(self, config):
        super().__init__(config=config)
        self.train_criterion = CriterionKGIN(config.train_loss)
        self.eval_criterion = CriterionKGIN(config.eval_loss)
        self.logger = Logger()
        self.scheduler_info = config.scheduler

    def run(self):
        pbar = tqdm.tqdm(range(self.start_epoch, self.num_epochs))
        for epoch in pbar:
            self.logger.reset_metric_item()
            start_time = time.time()
            print('run-start_time:',start_time)
            if not self.only_infer:
                self.train_one_epoch(epoch)
            self.run_test()
            self.logger.update_metric_item('train/epoch_runtime', (time.time() - start_time)/60)
            if epoch % self.config.weights_save_frequency == 0 and not self.debug and epoch > 150:
                self.save_model(epoch)
            if epoch == self.num_epochs - 1:
                self.save_model(epoch)
            if not self.debug:
                self.logger.wandb_log(epoch)

    def train_one_epoch(self, epoch):
        self.network.train()
        for i, (kspace, coilmaps, sampling_mask) in enumerate(self.train_loader):
            kspace,coilmaps,sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
            ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
            kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2) #[1,18,1,192]

            self.optimizer.zero_grad()
            adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

            with torch.cuda.amp.autocast(enabled=False):
                k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
                sampling_mask = sampling_mask.repeat_interleave(ref_kspace.shape[2], 2)
                ls = self.train_criterion(k_recon_2ch, torch.view_as_real(ref_kspace), im_recon, ref_img, kspace_mask=sampling_mask)

                self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())

            self.logger.update_metric_item('train/k_recon_loss', ls['k_recon_loss'].item()/len(self.train_loader))
            self.logger.update_metric_item('train/recon_loss', ls['photometric'].item()/len(self.train_loader))

    def run_test(self):
        out = torch.complex(torch.zeros([118, 18, 192, 192]), torch.zeros([118, 18, 192, 192])).to(device)
        self.network.eval()
        with torch.no_grad():
            for i, (kspace, coilmaps, sampling_mask) in enumerate(self.test_loader):
                kspace,coilmaps,sampling_mask = kspace.to(device), coilmaps.to(device), sampling_mask.to(device)
                ref_kspace, ref_img = multicoil2single(kspace, coilmaps)
                
                # wandb.log({"ref_kspace": ref_kspace})
                # wandb.log({"ref_kspace_abs": torch.abs(ref_kspace), "ref_kspace_angle": torch.angle(ref_kspace)})
                # wandb.log({"ref_kspace": ref_kspace}, json_encoder=MyEncoder)
                # wandb.log({"ref_kspace-shape": ref_kspace.shape})
                # wandb.log({"coilmaps": coilmaps}, json_encoder=MyEncoder)
                # wandb.log({"coilmaps :":coilmaps})
                # wandb.log({"ref_kspace": ref_kspace}, json_encoder=MyEncoder)
                # wandb.log({"coilmaps shape:":coilmaps.shape})
                # wandb.log({"sampling_mask:":sampling_mask})
                # wandb.log({"sampling_mask": sampling_mask}, json_encoder=MyEncoder)
                # wandb.log({"sampling_mask shape:":sampling_mask.shape})
                
                kspace = ref_kspace*torch.unsqueeze(sampling_mask, dim=2)

                k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask) # size of kspace and mask: [B, T, H, W]
                k_recon_2ch = k_recon_2ch[-1]

                kspace_complex = torch.view_as_complex(k_recon_2ch)
                sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)
                
                out[i] = kspace_complex

                ls = self.eval_criterion([kspace_complex], ref_kspace, im_recon, ref_img, kspace_mask=sampling_mask, mode='test')

                self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
                self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))
            print('...', out.shape, out.dtype)
            out = out.cpu().data.numpy()
            np.save('out_1121.npy', out)
            self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
            self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr'])




# wandb.log({"ref_kspace": ref_kspace}, json_encoder=MyEncoder)


    # def train_one_epoch(self, epoch):
    #     self.network.train()
    #     for i, batch in enumerate(self.train_loader):
    #         kspace, sampling_mask = [item.cuda() for item in batch[0]][:]
    #         ref = batch[1][0].cuda()

    #         self.optimizer.zero_grad()
    #         adjust_lr(self.optimizer, i/len(self.train_loader) + epoch, self.scheduler_info)

    #         with torch.cuda.amp.autocast(enabled=False):
    #             k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
    #             sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)
    #             ls = self.train_criterion(k_recon_2ch, torch.view_as_real(kspace), im_recon, ref, kspace_mask=sampling_mask)

    #             self.loss_scaler(ls['k_recon_loss_combined'], self.optimizer, parameters=self.network.parameters())

    #         self.logger.update_metric_item('train/k_recon_loss', ls['k_recon_loss'].item()/len(self.train_loader))
    #         self.logger.update_metric_item('train/recon_loss', ls['photometric'].item()/len(self.train_loader))

    # def run_test(self):
    #     self.network.eval()
    #     with torch.no_grad():
    #         for i, batch in enumerate(self.test_loader):
    #             kspace, sampling_mask = [item.cuda() for item in batch[0]][:]
    #             ref = batch[1][0].cuda()

    #             k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask) # size of kspace and mask: [B, T, H, W]
    #             k_recon_2ch = k_recon_2ch[-1]

    #             kspace_complex = torch.view_as_complex(k_recon_2ch)
    #             sampling_mask = sampling_mask.repeat_interleave(kspace.shape[2], 2)

    #             ls = self.eval_criterion([kspace_complex], kspace, im_recon, ref, kspace_mask=sampling_mask, mode='test')

    #             self.logger.update_metric_item('val/k_recon_loss', ls['k_recon_loss'].item()/len(self.test_loader))
    #             self.logger.update_metric_item('val/recon_loss', ls['photometric'].item()/len(self.test_loader))
    #             self.logger.update_metric_item('val/psnr', ls['psnr'].item()/len(self.test_loader))

    #         self.logger.update_best_eval_results(self.logger.get_metric_value('val/psnr'))
    #         self.logger.update_metric_item('train/lr', self.optimizer.param_groups[0]['lr'])
