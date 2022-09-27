import os
import math

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataloader import NIH_Dataset, train_data_transform, valid_data_transform, valid_data_transform_vit
from mode_resvit import resvit
from mode_vit import vit_base_patch16_224
from Unet import UNet

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler

from utils import iou_coef, Hyperparameter, create_lr_scheduler

def main():
    for epoch in range(epochs):
        epoch_loss = 0

        for img,mask, y in tqdm(trainloader):
            masks = mask.type(torch.FloatTensor).to(device)
            masks = torch.cat([masks, masks, masks], dim=1)
            images = img.type(torch.FloatTensor).to(device)
            y = y.type(torch.LongTensor).to(device)
            model.train()

            out = model(images)

            loss = criterion(out, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lrs == 1:
                scheduler.step()
            epoch_loss += loss / len(trainloader)

        tloss.append(epoch_loss.cpu().detach().numpy())

        with torch.no_grad():
            epoch_val_loss = 0
            # 计算验证集准确率
            imgs = []
            for img, mask, y in tqdm(validloader):
                images = img.type(torch.FloatTensor).to(device)
                masks = mask.type(torch.FloatTensor).to(device)
                masks = torch.cat([masks, masks, masks], dim=1)
                vy = y.type(torch.LongTensor).to(device)

                model.eval()
                val_output = model(images)
                val_loss = criterion(val_output, masks)
                imgs.append(images)
                epoch_val_loss += val_loss / len(validloader)

            vloss.append(epoch_val_loss.cpu().detach().numpy())
            torch.save(model.state_dict(), save_dir + '/best.pkl')

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f} - lr：{lr:.7f}\n"
            )
        #plt.figure(figsize=(12, 6))
        p = val_output[0].cpu().detach().numpy().transpose((1, 2, 0))
        plt.imshow(p)
        # myfig = plt.gcf()
        # myfig.savefig(save_dir + '/fig_acc.png')
        plt.show()

    print("保存成功")


if __name__ == '__main__':
    # 实例化SummaryWriter对象

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 超参数
    lrs = 1
    Pre = 0
    type = 0
    test = '/test'
    size, intype, modelname, schedular, epochs = Hyperparameter(size=2, Pre=Pre, model=0, lrs=lrs, type=type)
    epochs = 60
    save_dir = 'D:/ExpResult/train_segment/'

    train_covid_image_path = 'D:/Datasets/'+size+'/train/COVID/images/'
    train_normal_image_path = 'D:/Datasets/'+size+'/train/Normal/images/'
    train_pneumonia_image_path = 'D:/Datasets/'+size+'/train/Viral Pneumonia/images/'
    train_lung_opacity_image_path = 'D:/Datasets/'+size+'/train/Lung_Opacity/images/'
    # Paths to images
    val_covid_image_path = 'D:/Datasets/'+size+test+'/COVID/images/'
    val_normal_image_path = 'D:/Datasets/'+size+test+'/Normal/images/'
    val_pneumonia_image_path = 'D:/Datasets/'+size+test+'/Viral Pneumonia/images/'
    val_lung_opacity_image_path = 'D:/Datasets/'+size+test+'/Lung_Opacity/images/'

    # Paths to masks
    train_covid_mask_path = 'D:/Datasets/'+size+'/train/COVID/masks/'
    train_normal_mask_path = 'D:/Datasets/'+size+'/train/Normal/masks/'
    train_pneumonia_mask_path = 'D:/Datasets/'+size+'/train/Viral Pneumonia/masks/'
    train_lung_opacity_mask_path = 'D:/Datasets/'+size+'/train/Lung_Opacity/masks/'
    # Paths to masks
    val_covid_mask_path = 'D:/Datasets/'+size+test+'/COVID/masks/'
    val_normal_mask_path = 'D:/Datasets/'+size+test+'/Normal/masks/'
    val_pneumonia_mask_path = 'D:/Datasets/'+size+test+'/Viral Pneumonia/masks/'
    val_lung_opacity_mask_path = 'D:/Datasets/'+size+test+'/Lung_Opacity/masks/'

    # All paths to images and masks
    train_image_paths = [[train_covid_image_path + file for file in os.listdir(train_covid_image_path)]
                         + [train_normal_image_path + file for file in os.listdir(train_normal_image_path)]
                         + [train_pneumonia_image_path + file for file in os.listdir(train_pneumonia_image_path)]
                         + [train_lung_opacity_image_path + file for file in os.listdir(train_lung_opacity_image_path)]
                         ][0]
    train_mask_paths = [[train_covid_mask_path + file for file in os.listdir(train_covid_mask_path)]
                        + [train_normal_mask_path + file for file in os.listdir(train_normal_mask_path)]
                        + [train_pneumonia_mask_path + file for file in os.listdir(train_pneumonia_mask_path)]
                        + [train_lung_opacity_mask_path + file for file in os.listdir(train_lung_opacity_mask_path)]
                        ][0]
    val_image_paths = [[val_covid_image_path + file for file in os.listdir(val_covid_image_path)]
                       + [val_normal_image_path + file for file in os.listdir(val_normal_image_path)]
                       + [val_pneumonia_image_path + file for file in os.listdir(val_pneumonia_image_path)]
                       + [val_lung_opacity_image_path + file for file in os.listdir(val_lung_opacity_image_path)]
                       ][0]
    val_mask_paths = [[val_covid_mask_path + file for file in os.listdir(val_covid_mask_path)]
                      + [val_normal_mask_path + file for file in os.listdir(val_normal_mask_path)]
                      + [val_pneumonia_mask_path + file for file in os.listdir(val_pneumonia_mask_path)]
                      + [val_lung_opacity_mask_path + file for file in os.listdir(val_lung_opacity_mask_path)]
                      ][0]

    trainset = NIH_Dataset(train_image_paths, train_mask_paths, transform=train_data_transform, train=1, type=type)
    trainloader = DataLoader(trainset, batch_size=14, shuffle=True, num_workers=4, pin_memory=True)
    valset = NIH_Dataset(val_image_paths,  val_mask_paths, transform=valid_data_transform_vit, train=0, type=type)
    validloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

    model = UNet()
    pretrained_dict = torch.load(r'D:\ExpResult\train_segment\run20.pkl')
    net_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict)
    model.load_state_dict(net_state_dict)

    # 新的优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.AdamW(pg, lr=0.0001, weight_decay=5e-3)
    optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5e-4)
    if lrs == 1:
        scheduler = create_lr_scheduler(optimizer, len(trainloader), epochs,
                                       warmup=True, warmup_epochs=40)
    criterion = nn.BCELoss(weight=None)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    tacc = [0]
    tloss = [-1]
    vloss1 = [999]
    vacc = [0]
    vacc1 = [0]
    vloss = [-1]
    main()