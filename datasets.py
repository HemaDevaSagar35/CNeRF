import os
from cv2 import transform

import torch
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np

#This script is a copy of the file with same name from FeNeRF repo
COLOR_MAP = {
            0: [0, 0, 0], 
            1: [204, 0, 0],
            2: [76, 153, 0], 
            3: [204, 204, 0], 
            4: [51, 51, 255], 
            5: [204, 0, 204], 
            6: [0, 255, 255], 
            7: [255, 204, 204], 
            8: [102, 51, 0], 
            9: [255, 0, 0], 
            10: [102, 204, 0], 
            11: [255, 255, 0], 
            12: [0, 0, 153], 
            13: [0, 0, 204], 
            14: [255, 51, 153], 
            15: [0, 204, 204], 
            16: [0, 51, 0], 
            17: [255, 153, 51], 
            18: [0, 204, 0]}


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=4
    )
    return dataloader, 3


def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )

    return dataloader, 3


class FFHQDataset(Dataset):
    """
  
    """
    def __init__(self, dataset_path, img_size, return_label=True, **kwargs):
        img_base = 'ffhq_mask_img/*.png'
        label_base = 'ffhq_mask_mask/*.png'
        self.img_path = os.path.join(dataset_path, img_base)
        self.label_path = os.path.join(dataset_path,  label_base)
        self.transform_image = transforms.Compose(
                    [transforms.Resize(320), 
                    transforms.CenterCrop(256), 
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5]), 
                    # transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.Resize((img_size, img_size))])
        self.transform_label = transforms.Compose([
                    transforms.Resize(320, interpolation=PIL.Image.NEAREST), 
                    transforms.CenterCrop(256), 
                    transforms.ToTensor(), 
                    # transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.Resize((img_size, img_size), interpolation=PIL.Image.NEAREST)])
        self.data_img = sorted(glob.glob(self.img_path))
        self.data_label = sorted(glob.glob(self.label_path))
        
        self.return_label = return_label
        assert len(self.data_img) == len(self.data_label)
        
        self.color_map = COLOR_MAP

    def __len__(self):
        return len(self.data_img)

    def _mask_labels(self, mask_np):
        # label_size = len(self.color_map.keys())
        label_size = 12
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i, v in enumerate([0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 15, 17]):
            labels[i][mask_np==v] = 1.0
        return labels

    def __getitem__(self, index):
        # img = PIL.Image.open(self.data_img[index]).convert('RGB')
        img = PIL.Image.open(self.data_img[index])
        label = PIL.Image.open(self.data_label[index]).convert('L')
        
        ################################
        img = self.transform_image(img) # [-1, 1] after normalization
        label = self.transform_label(label)
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)

        label = self._mask_labels((label * 255.)[0]) 
        label = (label - 0.5) / 0.5 
        label = torch.tensor(label, dtype=torch.float)
        if not self.return_label:
            return img, 0
        return img, label, 0
