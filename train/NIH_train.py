import math

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from NIH_dataloader import train_data_transform, valid_data_transform, NIH_Dataset, processDF
from class_accuracy import class_accuracy, get_acc_data
from mode_our_1 import our_model

from mode_res18_vit import res18_vit_T
from mode_vit import vit_tiny_patch16_224
from model_efficientnetV2 import efficientnetv2_s
from model_resUnet import resUnet
from model_resnet import resnet18


def main():
    valid_loss_min = np.Inf
    for i in range(epochs):

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()
        for images, labels in tqdm(trainloader):
            images = images.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.FloatTensor)

            ps = model(images)
            loss = lossFunc(ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(trainloader)

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(validloader):
                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)

                ps = model(images)
                loss = lossFunc(ps, labels)
                valid_loss += loss.item()
            avg_valid_loss = valid_loss / len(validloader)

        schedular.step()

        if avg_valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(valid_loss_min,
                                                                                             avg_valid_loss))
            best=i
            torch.save(model.state_dict(), r'D:\ExpResult\pretrain_parameters\NIH_2vit16.pkl')
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss_min': avg_valid_loss
            }, 'NIH_Multi_label_test.pt')
            valid_loss_min = avg_valid_loss

        print("Epoch : {} Train Loss : {:.6f} ".format(i + 1, avg_train_loss))
        print("Epoch : {} Valid Loss : {:.6f} ".format(i + 1, avg_valid_loss))
        print("best: ",best)
    #torch.save(model.state_dict(), r'D:\ExpResult\pretrain_parameters\NIH_last.pkl')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    INPUT_DIR = 'D:/data/NIH_Chest_X_rays/'
    imgClasses = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                  'Pneumothorax']
    deletClasses = ['Consolidation', 'Emphysema', 'Hernia', 'Fibrosis', 'Pleural_Thickening', 'Edema']
    df = pd.read_csv(f'{INPUT_DIR}Data_Entry_2017.csv')
    df = df[df['Finding Labels'] != 'No Finding']
    for label in deletClasses:
        df = df[df['Finding Labels'].map(lambda x: False if label in x else True)]
    df = processDF(df)
    epochs = 200
    trains, tests = train_test_split(df, shuffle=True, test_size=0.2, random_state=69)
    # trains.to_excel('D:/project/NIH_Multi_label/result/traindata.xls')
    # tests.to_excel('D:/project/NIH_Multi_label/result/testdata.xls')

    # tests, vals = train_test_split(tests, shuffle=True, test_size=0.5, random_state=69)
    trainset = NIH_Dataset(trains, transform=train_data_transform)
    testset = NIH_Dataset(tests, transform=valid_data_transform)
    trainloader = DataLoader(trainset,
                             batch_size=80,
                             shuffle=True,
                             num_workers=6)

    validloader = DataLoader(testset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=4)

    model = vit_tiny_patch16_224(len(imgClasses)).to(device)

    # pretrained_dict = torch.load(r'D:\ExpResult\train\base\0img\Vit-16\scheduler\best.pkl')
    pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\NIH_1vit16.pkl')
    #pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet18-imagenet.pth')
    net_state_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head') if k in net_state_dict}
    net_state_dict.update(pretrained_dict)
    model.load_state_dict(net_state_dict)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.0001, weight_decay=0)

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    schedular = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    lossFunc = nn.BCEWithLogitsLoss().to(device)

    main()
    model.eval()
    valid_acc_list = class_accuracy(validloader, model)
    valid_acc = get_acc_data(imgClasses, valid_acc_list, 'D:/project/NIH_Multi_label/result/valid_acc_2vit16.csv')
    # train_acc_list = class_accuracy(trainloader, model)
    # train_acc = get_acc_data(imgClasses, train_acc_list, 'D:/project/NIH_Multi_label/result/train_acc.csv')
