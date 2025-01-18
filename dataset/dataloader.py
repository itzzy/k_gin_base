import torch
import torchvision
from utils import load_mask, ToTorchIO, multicoil2single
from dataset.transforms import *
from scipy.io import loadmat


# class CINE2DTBase(object):
#     def __init__(self, config, mode, transform):

#         self.mode = mode
#         self.dtype = config.dtype
#         self.transform = transform
#         self.current_epoch = 0

#         # subjs_csv = pd.read_csv(eval(f'config.{mode}_subjs_csv'))
#         # subjs_csv = pd.read_csv(eval(f'config.{mode}_subjs'),encoding='utf-8',errors='replace')
#         subjs_csv = pd.read_csv(eval(f'config.{mode}_subjs'))


#         self.data_names = [fname for fname in subjs_csv.filename]
#         self.data_names = [fname.split('.')[0] for fname in self.data_names]
#         self.data_paths = [os.path.join(config.data_root, f'{name}.h5') for name in self.data_names]
#         self.remarks = [fname for fname in subjs_csv.Remarks]

#         self.valid_slices = [np.arange(s_start, s_end) for s_start, s_end in zip(eval('subjs_csv.Valid_slice_start'), eval('subjs_csv.Valid_slice_end'))]
#         self.data_nPE = [nPE for nPE in subjs_csv.nPE]
#         self.data_nFE = [nFE for nFE in subjs_csv.nFE]
#         self.masks = {name: load_mask(config.mask_root, nPE, config.acc_rate[0]) for name, nPE in zip(self.data_names, self.data_nPE)}

#         self.data_list = []
#         self._build_data_list()

#     def __len__(self):
#         return len(self.data_list)

#     def _apply_transform(self, sample):
#         return self.transform(sample)

#     def get_current_epoch(self, epoch):
#         self.current_epoch = epoch

#     def __getitem__(self, idx):
#         sample = self._load_data(idx)
#         sample['epoch'] = self.current_epoch
#         if self.transform:
#             sample = self._apply_transform(sample)

#         # convert multi-coil to single-coil
#         sample[0][0], sample[1][0] = multicoil2single(sample[0][0], sample[0][2])
#         sample[0][1] = sample[0][1][0]
#         del sample[0][2]

#         return sample


# class CINE2DT(CINE2DTBase, torch.utils.data.Dataset):
#     def __init__(self, config, mode):
#         super().__init__(config=config, mode=mode, transform=None)
#         self.transform = torchvision.transforms.Compose(self.get_transform(config=config))

#     def _build_data_list(self):
#         for i, subj_name in enumerate(self.data_names):
#             for slc in self.valid_slices[i]:
#                 d = {'subj_name': subj_name,
#                      'data_path': self.data_paths[i],
#                      'slice': slc,
#                      'nPE': self.data_nPE[i],
#                      'remarks': self.remarks[i],
#                      }
#                 self.data_list.append(d)

#     def _load_data(self, idx):
#         data = self.data_list[idx]
#         subj_name = data['subj_name']
#         # print(f' Loading Data {subj_name}')
#         slice = data['slice']
#         with h5py.File(data['data_path'], 'r', swmr=True, libver='latest') as ds:
#             d = {
#                  'kspace': ds['kSpace'][slice].astype(eval(f'np.{self.dtype}')).transpose(0, 1, 3, 2),
#                  'smaps': ds['dMap'][slice].astype(eval(f'np.{self.dtype}')).transpose(0, 1, 3, 2),
#                  'reference': ds['dImgC'][slice].astype(eval(f'np.{self.dtype}')).transpose(0, 1, 3, 2).squeeze(),
#                  'subj_name': subj_name,
#                  'nPE': data['nPE'],
#                  'slice': slice,
#                  'mask': self.masks[subj_name],
#                  'remarks': data['remarks']
#                  }
#         return d

#     def get_transform(self, config):
#         assert self.mode in ('train', 'val', 'infer')

#         data_transforms = []
#         if self.mode == 'train':
#             data_transforms.append(LoadMask(config.mask_pattern, config.acc_rate, config.mask_root))
#         data_transforms.append(Normalize(mode='3D', scale=1.0))
#         data_transforms.append(ToNpDtype([('reference', np.complex64), ('kspace', np.complex64),
#                                           ('smaps', np.complex64), ('mask', np.float32), ]))
#         data_transforms.append(ExtractTimePatch(config.training_patch_time, [('reference', 0), ('kspace', 1), ('mask', 1)], mode=self.mode))
#         input_convert_list = ['kspace', 'mask', 'smaps']
#         data_transforms.append(ToTorchIO(input_convert_list, ['reference']))

#         return data_transforms


# 定义了一个名为 CINE2DT 的类，用于加载和处理 CINE2D 数据集,它继承自 torch.utils.data.Dataset。
# 这意味着该类是一个数据集类，可以被 PyTorch 的数据加载器使用。
class CINE2DT(torch.utils.data.Dataset):
    # config: 一个配置对象，包含了数据集的路径和相关参数。
    # mode: 表示数据集的模式，可以是 'train'、'val' 或 'test'。
    def __init__(self, config, mode):
        super(CINE2DT, self).__init__()
        self.mode = mode
        # 根据数据集模式，加载对应的标签数据和卷积灵敏度图 (Coil Sensitivity Maps)。
        if mode == 'train':
            self._label = np.load(os.path.join(config.train_subjs)).astype(np.complex64)
            self._csm = np.load(os.path.join(config.train_maps)).astype(np.complex64)
        elif mode == 'val':
            self._label = np.load(os.path.join(config.val_subjs)).astype(np.complex64)
            self._csm = np.load(os.path.join(config.val_maps)).astype(np.complex64)
        else:
            raise NotImplementedError
        # 从指定的路径加载一个 MATLAB 文件，该文件包含了采样掩码 (Mask)。
        C =loadmat(config.mask_root)
        # 从加载的 MATLAB 文件中获取 mask 数据，并将其保存到 self.mask 属性中。
        #  prep_input-mask-shape: (192, 18)
        self.mask = C['mask'][:]
        # 对 self.mask 进行转置操作。
        self.mask = np.transpose(self.mask,[1,0])

    def __getitem__(self, index):
        # 获取第 index 个样本的 k-space 数据。
        kspace = self._label[index, :]
        # 获取第 index 个样本的卷积灵敏度图。
        coilmaps = self._csm[index, :]
        # 获取采样掩码，并将其转换为 np.int32 类型。
        sampling_mask = self.mask.astype(np.int32)
        return kspace, coilmaps,sampling_mask

    def __len__(self):
        # 返回标签数据的第一个维度的大小，即数据集的大小。
        return self._label.shape[0]

'''
CINE2DT 类是一个数据集类，用于加载和处理 CINE2D 数据集。它包含了 __init__、__getitem__ 和 __len__ 函数，
分别用于初始化数据集对象、返回单个样本和返回数据集大小。该类可以被 PyTorch 的数据加载器使用，用于训练和验证深度学习模型。
'''