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
from nihloader import NIH_Dataset, train_data_transform, valid_data_transform, valid_data_transform_vit
from mode_resvit1 import resvit, resvit_base, resvit_large
from mode_vit import vit_base_patch16_224
from model_convnext import convnext_tiny
from model_densenet import densenet121
from model_resF import resF18
from model_resUnet import resUnet
from model_resnet import resnet50, resnet101, resnet18

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T

from model_swint import swin_tiny_patch4_window7_224
from utils import iou_coef, Hyperparameter, create_lr_scheduler


def main():
    tags = ["loss", "loss1", "loss2", "accuracy", "learning_rate", "val_acc"]
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0

        epoch_accuracy = 0
        prediction = []
        true_labels = []
        resize = T.Resize(size=(32, 32))
        for img, mask, y in tqdm(trainloader):

            masks = mask.type(torch.FloatTensor).to(device)
            masks = resize(masks)
            images = img.type(torch.FloatTensor).to(device)
            y = y.type(torch.LongTensor).to(device)
            model.train()
            out, out2 = model(images)
            loss1 = criterion(out, y)
            loss2 = criterion1(out2, masks)
            loss = loss1 + 0.5 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lrs == 1:
                scheduler.step()
            acc = (out.argmax(dim=1) == y).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
            epoch_loss1 += loss1 / len(trainloader)
            epoch_loss2 += loss2 / len(trainloader)
            prediction += out.argmax(dim=1).tolist()
            true_labels += y.tolist()

        tacc.append(epoch_accuracy.cpu().detach().numpy())
        tloss.append(epoch_loss.cpu().detach().numpy())
        confusion_mtx = confusion_matrix(true_labels, prediction)
        print("loss1:", epoch_loss1, "loss2:", epoch_loss2)
        with torch.no_grad():
            prediction = []
            true_labels = []
            epoch_val_accuracy = 0
            epoch_val_loss1 = 0
            epoch_val_loss2 = 0
            epoch_val_loss = 0
            # 计算验证集准确率
            imgs = []
            for img, mask, y in tqdm(validloader):
                images = img.type(torch.FloatTensor).to(device)
                masks = mask.type(torch.FloatTensor).to(device)
                masks = resize(masks)
                vy = y.type(torch.LongTensor).to(device)

                model.eval()
                val_output, val_output2 = model(images)
                val_loss1 = criterion(val_output, vy)
                val_loss2 = criterion1(val_output2, masks)

                val_loss = val_loss1+0.5*val_loss2
                imgs.append(images)
                acc = (val_output.argmax(dim=1) == vy).float().mean()
                epoch_val_accuracy += acc / len(validloader)
                epoch_val_loss += val_loss / len(validloader)
                epoch_val_loss1 += val_loss1 / len(validloader)
                epoch_val_loss2 += val_loss2 / len(validloader)
                prediction += val_output.argmax(dim=1).tolist()
                true_labels += vy.tolist()
            p1 = imgs[0][0].cpu().detach().numpy().transpose((1, 2, 0))

            vloss.append(epoch_val_loss.cpu().detach().numpy())
            vacc.append(epoch_val_accuracy.cpu().detach().numpy())
            if max(vacc1) < epoch_val_accuracy:
                print("最优epoch: ", epoch + 1)
                best = epoch+1
                best_acc = epoch_val_accuracy
                torch.save(model.state_dict(), save_dir+'/best.pkl')

            if min(vloss1) > epoch_val_loss:
                estop =0
            else:
                estop += 1
                print("Counter {} of 60".format(estop))
                if estop > 200:
                    print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ",
                          epoch_val_accuracy, "...")
                    break
            vacc1.append(epoch_val_accuracy)
            vloss1.append(epoch_val_loss)
            tb_writer.add_scalars(tags[0], {'train_loss': epoch_loss.cpu().detach().numpy(),
                                            'val_loss': epoch_val_loss.cpu().detach().numpy()}, epoch)
            tb_writer.add_scalars(tags[1], {'train_loss': epoch_loss1.cpu().detach().numpy(),
                                            'val_loss': epoch_val_loss1.cpu().detach().numpy()}, epoch)
            tb_writer.add_scalars(tags[2], {'train_loss': epoch_loss2.cpu().detach().numpy(),
                                            'val_loss': epoch_val_loss2.cpu().detach().numpy()}, epoch)
            tb_writer.add_scalars(tags[3], {'train_acc': epoch_accuracy.cpu().detach().numpy(),
                                            'val_acc': epoch_val_accuracy.cpu().detach().numpy()}, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        confusion_mtx1 = confusion_matrix(true_labels, prediction)
        print("epoch_val_loss1:", epoch_val_loss1, "epoch_val_loss2:", epoch_val_loss2)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - lr：{lr:.7f}\n"
            )
        #print("train", confusion_mtx)
        print("val", confusion_mtx1)
        print(best, best_acc)
    torch.save(model.state_dict(), save_dir + '/last.pkl')
    plt.figure(figsize=(12, 6))
    plt.plot(list(map(float, vacc[:])), 'g', label='val_acc')
    plt.plot(list(map(float, tacc[:])), 'r', label='train_acc')
    plt.xticks(list(range(0, epoch+1, 20)))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    myfig = plt.gcf()
    myfig.savefig(save_dir + '/fig_acc.png')
    plt.show()
    output_excel = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    output_excel['train_loss'] = tloss
    output_excel['train_acc'] = tacc
    output_excel['test_loss'] = vloss
    output_excel['test_acc'] = vacc

    output = pd.DataFrame(output_excel)
    output.to_excel(save_dir + '/output_excel.xlsx', index=False)
    print("保存成功")


if __name__ == '__main__':
    # 实例化SummaryWriter对象

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 超参数
    lrs = 1
    Pre = 1
    type = 0
    test = '/test'
    size, intype, modelname, schedular, epochs = Hyperparameter(size=3, Pre=Pre, model=3, lrs=lrs, type=type)
    save_dir = 'D:/ExpResult/train/' + size + intype + modelname + schedular
    tb_writer = SummaryWriter(log_dir=save_dir + "/runs/chest_experiment1")
    hyperparameter_excel = {'data_size': [], 'intype': [], 'modelname': [], 'schedular': []}
    hyperparameter_excel['data_size'] = size
    hyperparameter_excel['intype'] = intype
    hyperparameter_excel['modelname'] = modelname
    hyperparameter_excel['schedular'] = schedular
    hyperparameter = pd.DataFrame(hyperparameter_excel,index=[0])
    hyperparameter.to_excel(save_dir + '/hyperparameter_excel.xlsx', index=False)

    train_Atelectasis_image_path = 'D:/Datasets/' + size + '/train/Atelectasis/images/'
    train_Cardiomegaly_image_path = 'D:/Datasets/' + size + '/train/Cardiomegaly/images/'
    train_Effusion_image_path = 'D:/Datasets/' + size + '/train/Effusion/images/'
    train_Infiltrate_image_path = 'D:/Datasets/' + size + '/train/Infiltrate/images/'
    train_Mass_image_path = 'D:/Datasets/' + size + '/train/Mass/images/'
    train_Nodule_image_path = 'D:/Datasets/' + size + '/train/Nodule/images/'
    train_Pneumonia_image_path = 'D:/Datasets/' + size + '/train/Pneumonia/images/'
    train_Pneumothorax_image_path = 'D:/Datasets/' + size + '/train/Pneumothorax/images/'

    # Paths to masks
    train_Atelectasis_mask_path = 'D:/Datasets/' + size + '/train/Atelectasis/masks/'
    train_Cardiomegaly_mask_path = 'D:/Datasets/' + size + '/train/Cardiomegaly/masks/'
    train_Effusion_mask_path = 'D:/Datasets/' + size + '/train/Effusion/masks/'
    train_Infiltrate_mask_path = 'D:/Datasets/' + size + '/train/Infiltrate/masks/'
    train_Mass_mask_path = 'D:/Datasets/' + size + '/train/Mass/masks/'
    train_Nodule_mask_path = 'D:/Datasets/' + size + '/train/Nodule/masks/'
    train_Pneumonia_mask_path = 'D:/Datasets/' + size + '/train/Pneumonia/masks/'
    train_Pneumothorax_mask_path = 'D:/Datasets/' + size + '/train/Pneumothorax/masks/'
    # Paths to masks
    test_Atelectasis_image_path = 'D:/Datasets/' + size + '/test/Atelectasis/images/'
    test_Cardiomegaly_image_path = 'D:/Datasets/' + size + '/test/Cardiomegaly/images/'
    test_Effusion_image_path = 'D:/Datasets/' + size + '/test/Effusion/images/'
    test_Infiltrate_image_path = 'D:/Datasets/' + size + '/test/Infiltrate/images/'
    test_Mass_image_path = 'D:/Datasets/' + size + '/test/Mass/images/'
    test_Nodule_image_path = 'D:/Datasets/' + size + '/test/Nodule/images/'
    test_Pneumonia_image_path = 'D:/Datasets/' + size + '/test/Pneumonia/images/'
    test_Pneumothorax_image_path = 'D:/Datasets/' + size + '/test/Pneumothorax/images/'

    # Paths to masks
    test_Atelectasis_mask_path = 'D:/Datasets/' + size + '/test/Atelectasis/masks/'
    test_Cardiomegaly_mask_path = 'D:/Datasets/' + size + '/test/Cardiomegaly/masks/'
    test_Effusion_mask_path = 'D:/Datasets/' + size + '/test/Effusion/masks/'
    test_Infiltrate_mask_path = 'D:/Datasets/' + size + '/test/Infiltrate/masks/'
    test_Mass_mask_path = 'D:/Datasets/' + size + '/test/Mass/masks/'
    test_Nodule_mask_path = 'D:/Datasets/' + size + '/test/Nodule/masks/'
    test_Pneumonia_mask_path = 'D:/Datasets/' + size + '/test/Pneumonia/masks/'
    test_Pneumothorax_mask_path = 'D:/Datasets/' + size + '/test/Pneumothorax/masks/'

    # All paths to images and masks
    train_image_paths = [[train_Atelectasis_image_path + file for file in os.listdir(train_Atelectasis_image_path)]
                        + [train_Cardiomegaly_image_path + file for file in os.listdir(train_Cardiomegaly_image_path)]
                        + [train_Effusion_image_path + file for file in os.listdir(train_Effusion_image_path)]
                        + [train_Infiltrate_image_path + file for file in os.listdir(train_Infiltrate_image_path)]
                        + [train_Mass_image_path + file for file in os.listdir(train_Mass_image_path)]
                        + [train_Nodule_image_path + file for file in os.listdir(train_Nodule_image_path)]
                        + [train_Pneumonia_image_path + file for file in os.listdir(train_Pneumonia_image_path)]
                        + [train_Pneumothorax_image_path + file for file in os.listdir(train_Pneumothorax_image_path)]
                        ][0]
    train_mask_paths = [[train_Atelectasis_mask_path + file for file in os.listdir(train_Atelectasis_mask_path)]
                       + [train_Cardiomegaly_mask_path + file for file in os.listdir(train_Cardiomegaly_mask_path)]
                       + [train_Effusion_mask_path + file for file in os.listdir(train_Effusion_mask_path)]
                       + [train_Infiltrate_mask_path + file for file in os.listdir(train_Infiltrate_mask_path)]
                       + [train_Mass_mask_path + file for file in os.listdir(train_Mass_mask_path)]
                       + [train_Nodule_mask_path + file for file in os.listdir(train_Nodule_mask_path)]
                       + [train_Pneumonia_mask_path + file for file in os.listdir(train_Pneumonia_mask_path)]
                       + [train_Pneumothorax_mask_path + file for file in os.listdir(train_Pneumothorax_mask_path)]
                       ][0]
    test_image_paths = [[test_Atelectasis_image_path + file for file in os.listdir(test_Atelectasis_image_path)]
                        + [test_Cardiomegaly_image_path + file for file in os.listdir(test_Cardiomegaly_image_path)]
                        + [test_Effusion_image_path + file for file in os.listdir(test_Effusion_image_path)]
                        + [test_Infiltrate_image_path + file for file in os.listdir(test_Infiltrate_image_path)]
                        + [test_Mass_image_path + file for file in os.listdir(test_Mass_image_path)]
                        + [test_Nodule_image_path + file for file in os.listdir(test_Nodule_image_path)]
                        + [test_Pneumonia_image_path + file for file in os.listdir(test_Pneumonia_image_path)]
                        + [test_Pneumothorax_image_path + file for file in os.listdir(test_Pneumothorax_image_path)]
                        ][0]
    test_mask_paths = [[test_Atelectasis_mask_path + file for file in os.listdir(test_Atelectasis_mask_path)]
                        + [test_Cardiomegaly_mask_path + file for file in os.listdir(test_Cardiomegaly_mask_path)]
                        + [test_Effusion_mask_path + file for file in os.listdir(test_Effusion_mask_path)]
                        + [test_Infiltrate_mask_path + file for file in os.listdir(test_Infiltrate_mask_path)]
                        + [test_Mass_mask_path + file for file in os.listdir(test_Mass_mask_path)]
                        + [test_Nodule_mask_path + file for file in os.listdir(test_Nodule_mask_path)]
                        + [test_Pneumonia_mask_path + file for file in os.listdir(test_Pneumonia_mask_path)]
                        + [test_Pneumothorax_mask_path + file for file in os.listdir(test_Pneumothorax_mask_path)]
                        ][0]

    trainset = NIH_Dataset(train_image_paths, train_mask_paths, transform=train_data_transform, train=1, type=type)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valset = NIH_Dataset(test_image_paths,  test_mask_paths, transform=valid_data_transform_vit, train=0, type=type)
    validloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

    if modelname == '/Resvit_Pretrain':
        model = resvit(num_classes=8)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet34-imagenet.pth')
        #pretrained_dict = torch.load(r'D:\ExpRelsult\pretrain_parameters\best18_pretrain.pkl')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    elif modelname == '/Resvit_base_Pretrain':
        model = resvit_base(num_classes=8)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet18-imagenet.pth')
        #pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\best34_pretrain.pkl')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    elif modelname == '/Resvit_large_Pretrain':
        model = resvit_large(num_classes=8)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet50-imagenet.pth')
        #pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\best50_pretrain.pkl')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    elif modelname == '/Resvit':
        model = resvit(num_classes=8)
    elif modelname == '/Resvit_base':
        model = resvit_base(num_classes=8)
    elif modelname == '/Resvit_large':
        model = resvit_large(num_classes=8)
    if Pre == 1:
        net_state_dict.update(pretrained_dict)
        model.load_state_dict(net_state_dict)

    # 新的优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.AdamW(pg, lr=0.0001, weight_decay=5e-3)
    optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5e-4)
    if lrs == 1:
        scheduler = create_lr_scheduler(optimizer, len(trainloader), epochs,
                                       warmup=True, warmup_epochs=10)
    criterion = nn.CrossEntropyLoss(weight=None)
    criterion1 = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()
    tacc = [0]
    tloss = [-1]
    vloss1 = [999]
    vacc = [0]
    vacc1 = [0]
    vloss = [-1]
    main()