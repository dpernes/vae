import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class ImgDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.imgs_list = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(*(self.path, self.imgs_list[index])))

        if self.transform is not None:
            img = self.transform(img)

        return img
