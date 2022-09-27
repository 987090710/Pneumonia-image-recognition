import os

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


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.03, sh=0.17, r1=0.3, mean=(0, 0, 0)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


train_data_transform = T.Compose([
    #T.RandomResizedCrop(256),
    T.RandomCrop(256),
    #T.GaussianBlur(3),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=(-15, +15)),
    #T.RandomAffine(translate=(0.2, 0.2), degrees=0),

    T.ToTensor(),
])
valid_data_transform = T.Compose([
    T.ToTensor(),
])
valid_data_transform_vit = T.Compose([
    # T.CenterCrop((256, 256)),
    T.ToTensor(),
])


def Histeq(im):
    imhist, bins = histogram(im.flatten())
    cdf = imhist.cumsum()
    cdf = cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape)


class NIH_Dataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None, train=1, type=0, batch_size=16):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
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
        mask_file = self.mask_paths[idx]
        mask = Image.open(mask_file).convert('L')
        split = img_file.split('/')
        class_name = split[-3]
        if class_name == "bacteria":
            label = np.array(0).astype(float)
        elif class_name == "normal":
            label = np.array(1).astype(float)
        elif class_name == "virus":
            label = np.array(2).astype(float)
        totensor = T.ToTensor()
        topil = T.ToPILImage()
        resize = T.Resize(size=(256, 256))

        img1 = resize(img1)
        img1 = totensor(img1)
        mask = totensor(mask)
        cat = torch.cat([img1, mask], dim=0)
        cat = topil(cat)
        if self.transform:

            # img = self.transform(img)
            # mask = self.transform(mask)
            cat = self.transform(cat)

        img = cat[0, :, :].unsqueeze(0)
        mask = cat[1, :, :].unsqueeze(0)

        r = np.random.rand(2)
        if (r.argmax() == 0):
            R = 0.75
        elif (r.argmax() == 1):
            R = 1.25
        a = np.random.rand(256, 256)
        b = 0.05 * random()

        a = a > b
        g_c = (1 - img) ** R
        c_g = (1 - img ** R)
        gamma = (img) ** R
        complement = (1 - img)
        masks = (mask * img)
        nose = torch.tensor(a.astype(int)) * img
        mask_gamma = (mask * img) ** R
        mask_complement = ((1 - (mask) * img) * (mask))
        histeq = torch.tensor(Histeq(img))
        local_nose = (1 - mask) * img + torch.tensor(a.astype(int)) * mask * img
        local_histeq = (1 - mask) * img + mask * histeq
        local_gamma = (1 - mask) * img + (0.5 * random() + 0.5) * (mask * img) ** R
        local_complement = (1 - mask) * img + (0.5 * random() + 0.5) * ((1 - (mask) * img) * (mask))
        local_masks = (1 - mask) * img + (0.5 * random() + 0.5) * mask * img


        if self.train == 1:
            if self.type == 0:
                img1 = torch.cat([img, img, img], dim=0)
                # img1 = self.RE(img1)
               #img1 = self.T(img1)
            if self.type == 1:
                A = np.random.rand(4)
                if (A.argmax() == 0):
                    img1 = img
                elif (A.argmax() == 1):
                    img1 = gamma
                elif (A.argmax() == 2):
                    img1 = complement
                elif (A.argmax() == 3):
                    img1 = nose
                img1 = torch.cat([img1, img1, img1], dim=0)
                img1 = self.RE(img1)
                # for i in range(2):
                #     A = np.random.rand(4)
                #     if (A.argmax() == 0):
                #         img = img
                #     elif (A.argmax() == 1):
                #         img = gamma
                #     elif (A.argmax() == 2):
                #         img = complement
                #     elif (A.argmax() == 3):
                #         img = (gamma + complement + img) / 3
                #     img1 = torch.cat([img1, img], dim=0)

                # img1 = self.T(img1)
            if self.type == 2:
                img1 = torch.cat([masks, masks, masks], dim=0)
                #img1 = self.RE(img1)
            if self.type == 3:
                A = np.random.rand(4)
                if (A.argmax() == 0):
                    img1 = img
                elif (A.argmax() == 1):
                    img1 = local_gamma
                elif (A.argmax() == 2):
                    img1 = local_complement
                elif (A.argmax() == 3):
                    img1 = local_nose
                # for i in range(2):
                #     A = np.random.rand(4)
                #     if (A.argmax() == 0):
                #         img = local_masks
                #     elif (A.argmax() == 1):
                #         img = local_gamma
                #     elif (A.argmax() == 2):
                #         img = local_complement
                #     elif (A.argmax() == 3):
                #         img = (local_gamma + local_complement + local_masks) / 3
                #     elif (A.argmax() == 4):
                #         img = local_nose
                #     img1 = torch.cat([img1, img], dim=0)
                # img1 = self.T(img1)
                img1 = torch.cat([img1, img1, img1], dim=0)
                img1 = self.RE(img1)
            if self.type == 4:
                img1 = torch.cat([complement, complement, complement], dim=0)
                img1 = self.RE(img1)
            if self.type == 5:
                img1 = torch.cat([gamma, gamma, gamma], dim=0)
                img1 = self.RE(img1)
            if self.type == 6:
                img1 = torch.cat([histeq, histeq, histeq], dim=0)
                #img1 = self.RE(img1)
            if self.type == 7:
                img1 = torch.cat([img, gamma, complement, g_c, c_g, histeq, local_histeq, local_gamma, local_complement],
                                 dim=0)
                # img1 = self.RE(img1)

        elif self.train == 0:
            if self.type == 0:
                img1 = torch.cat([img, img, img], dim=0)
            # if self.type == 1:
            #     img1 = torch.cat([img, img, img], dim=0)
            if self.type == 2:
                img1 = torch.cat([masks, masks, masks], dim=0)
            # if self.type == 3:
            #     img1 = torch.cat([masks, masks, masks], dim=0)
            if self.type == 4:
                img1 = torch.cat([complement, complement, complement], dim=0)
            if self.type == 5:
                img1 = torch.cat([gamma, gamma, gamma], dim=0)
            if self.type == 6:
                img1 = torch.cat([histeq, histeq, histeq], dim=0)
            if self.type == 7:
                img1 = torch.cat([img, gamma, complement, g_c, c_g, histeq, local_histeq, local_gamma, local_complement],
                                 dim=0)

        return img1, mask, label



