# -- coding: utf-8 --
from random import random
from glob import glob
from pylab import *
import math
import torch
from torchtoolbox.transform import Cutout
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from dataloader import RandomErasing

valid_data_transform = T.Compose([
    T.ToTensor(),
])

class NIH_Dataset(Dataset):

    def __init__(self, image_paths, transform=None, train=1, type=0):
        self.image_paths = image_paths
        self.transform = transform
        self.train = train
        self.type = type
        self.T = T.RandomResizedCrop(256)
        self.RE = RandomErasing(probability=0.5)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_file = self.image_paths[idx]
        img1 = Image.open(img_file).convert('L')
        split = img_file.split('/')
        class_name = split[-3]
        image_name = split[-1]
        if class_name == "Atelectasis":
            label = np.array(0).astype(float)
        elif class_name == "Cardiomegaly":
            label = np.array(1).astype(float)
        elif class_name == "Effusion":
            label = np.array(2).astype(float)
        elif class_name == "Infiltrate":
            label = np.array(3).astype(float)
        elif class_name == "Mass":
            label = np.array(4).astype(float)
        elif class_name == "Nodule":
            label = np.array(5).astype(float)
        elif class_name == "Pneumonia":
            label = np.array(6).astype(float)
        elif class_name == "Pneumothorax":
            label = np.array(7).astype(float)
        resize = T.Resize(size=(256, 256))

        img1 = resize(img1)

        if self.transform:
            img = self.transform(img1)
        img1 = torch.cat([img, img, img], dim=0)


        return img1, class_name, image_name