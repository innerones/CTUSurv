import numpy
import torch
import config
from tqdm import tqdm
import numpy as np
from timm.data import Mixup
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from .samplers import SubsetRandomSampler
# NLST_path = '/home/nclei_seg/processed_data/'
class subDataset(Dataset):
    def __init__(self, nuclei, image, label):
        self.nuclei = nuclei
        self.image = image
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # nuclei = torch.tensor(self.nuclei[index])
        # image = torch.tensor(self.image[index])
        # label = torch.tensor(self.label[index])
        nuclei = self.nuclei[index].clone().detach()
        image = self.image[index].clone().detach()
        label = self.label[index].clone().detach()
        return nuclei, image, label



def build_loader(config):
    path = config.DATA.DATA_PATH
    x_data = np.load(path + 'normalized_nuclei.npy', allow_pickle=True)
    y_data = np.load(path + 'label.npy').astype(int)
    image_data = np.load(path + 'image_features.npy', allow_pickle=True)
    image = torch.from_numpy(image_data)
    nuclei_mask_data = np.load('/home/single_nuclei/nucleimask.npz')
    # x_data = np.load(NLST_path + 'normalized_nuclei.npy', allow_pickle=True)
    # y_data = np.load(NLST_path + 'label.npy').astype(int)
    # image_data = np.load(NLST_path + 'image_features.npy', allow_pickle=True)
    # image = torch.from_numpy(image_data)
    # nuclei_mask_data = np.load(NLST_path+'nucleimask.npz')
    nuclei_mask = nuclei_mask_data['nuclei_mask']
    nuclei_mask = torch.from_numpy(nuclei_mask)
    # x_data = np.nan_to_num(x_data)
    x = torch.from_numpy(x_data)
    x = x[:, :, :, (0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22)]
    x = torch.cat((nuclei_mask, x), 3)
    y = torch.from_numpy(y_data)
    x = x[:, :, :, 0:config.MODEL.SWIN.EMBED_DIM+1]
    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)
    # final_dataset = TensorDataset(x, y)
    final_dataset = subDataset(x, image, y)

    dataset_train, dataset_val, dataset_test = random_split(dataset=final_dataset, lengths=[69, 69, 548])
    # dataset_train, dataset_val, dataset_test = random_split(dataset=final_dataset, lengths=[359, 45, 45])
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        shuffle=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    dataset_loader_test = DataLoader(
        dataset_test,
        batch_size=len(dataset_test),
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, dataset_loader_test, mixup_fn
