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
from mode_efficient_vit import efficient_vit_T
from mode_our_1 import our_model

from mode_res18_vit import res18_vit_T, res18_vit8_T

from mode_vit import vit_tiny_patch16_224, vit_base_patch32
from model_convnext import convnext_tiny
from model_densenet import densenet121
from model_efficientnetV2 import efficientnetv2_s

from model_resnet import resnet50,  resnet18,  resnext50_32x4d1
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler

from utils import iou_coef, Hyperparameter, create_lr_scheduler


def main():
    tags = ["loss", "accuracy", "learning_rate", "val_acc"]
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        prediction = []
        true_labels = []
        resize = T.Resize(size=(224, 224))
        for img,mask, y in tqdm(trainloader):

            images = img.type(torch.FloatTensor).to(device)
            y = y.type(torch.LongTensor).to(device)
            model.train()
            # p1 = images[0].cpu().detach().numpy().transpose((1, 2, 0))
            # plt.imshow(p1[:, :, :], cmap=plt.get_cmap('gray'))
            # plt.show()

            out = model(images)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lrs == 1:
                scheduler.step()
            acc = (out.argmax(dim=1) == y).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
            prediction += out.argmax(dim=1).tolist()
            true_labels += y.tolist()

        tacc.append(epoch_accuracy.cpu().detach().numpy())
        tloss.append(epoch_loss.cpu().detach().numpy())
        confusion_mtx = confusion_matrix(true_labels, prediction)

        with torch.no_grad():
            prediction = []
            true_labels = []
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            # 计算验证集准确率
            imgs = []
            for img,mask, y in tqdm(validloader):
                images = img.type(torch.FloatTensor).to(device)

                vy = y.type(torch.LongTensor).to(device)

                model.eval()
                val_output = model(images)
                val_loss = criterion(val_output, vy)

                imgs.append(images)
                acc = (val_output.argmax(dim=1) == vy).float().mean()
                epoch_val_accuracy += acc / len(validloader)
                epoch_val_loss += val_loss / len(validloader)
                prediction += val_output.argmax(dim=1).tolist()
                true_labels += vy.tolist()
            p1 = imgs[0][0].cpu().detach().numpy().transpose((1, 2, 0))

            vloss.append(epoch_val_loss.cpu().detach().numpy())
            vacc.append(epoch_val_accuracy.cpu().detach().numpy())
            if max(vacc1) < epoch_val_accuracy:
                print("最优epoch: ", epoch + 1)
                best = epoch+1
                best_acc = epoch_val_accuracy
                # torch.save(model.state_dict(), save_dir+'/best.pkl')

            if min(vloss1) > epoch_val_loss:
                estop =0
            else:
                estop += 1
                if lrs == 0:
                    print("Counter {} of xxx".format(estop))
                    # if estop > 24 and epoch>80:
                    #     print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ",
                    #           epoch_val_accuracy, "...")
                    #     break
                elif lrs == 1:
                    print("Counter {} of xxx".format(estop))
                    # if estop > 24 and epoch>80:
                    #     print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ",
                    #           epoch_val_accuracy, "...")
                    #     break
            vacc1.append(epoch_val_accuracy)
            vloss1.append(epoch_val_loss)
            tb_writer.add_scalars(tags[0], {'train_loss': epoch_loss.cpu().detach().numpy(),
                                            'val_loss': epoch_val_loss.cpu().detach().numpy()}, epoch)
            tb_writer.add_scalars(tags[1], {'train_acc': epoch_accuracy.cpu().detach().numpy(),
                                            'val_acc': epoch_val_accuracy.cpu().detach().numpy()}, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        confusion_mtx1 = confusion_matrix(true_labels, prediction)

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - lr：{lr:.10f}\n"
            )
        print("train",confusion_mtx)
        print("val",confusion_mtx1)
        print(best, best_acc)
    # torch.save(model.state_dict(), save_dir + '/last.pkl')
    # plt.figure(figsize=(12, 6))
    # plt.plot(list(map(float, vacc[:])), 'g', label='val_acc')
    # plt.plot(list(map(float, tacc[:])), 'r', label='train_acc')
    # plt.xticks(list(range(0, epoch+1, 20)))
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.grid()
    # myfig = plt.gcf()
    # myfig.savefig(save_dir + '/fig_acc.png')
    # plt.show()
    # output_excel = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    # output_excel['train_loss'] = tloss
    # output_excel['train_acc'] = tacc
    # output_excel['test_loss'] = vloss
    # output_excel['test_acc'] = vacc
    #
    # output = pd.DataFrame(output_excel)
    # output.to_excel(save_dir + '/output_excel.xlsx', index=False)
    # print("保存成功")


if __name__ == '__main__':
    # 实例化SummaryWriter对象

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 超参数
    lrs = 1
    Pre = 1
    type = 0
    test = '/test'
    size, intype, modelname, schedular, epochs = Hyperparameter(size=2, Pre=Pre, model=10, lrs=lrs, type=type)
    save_dir = 'D:/ExpResult/train/' + size + intype + modelname + schedular
    tb_writer = SummaryWriter(log_dir="./runs" + modelname)
    hyperparameter_excel = {'data_size': [], 'intype': [], 'modelname': [], 'schedular': []}
    hyperparameter_excel['data_size'] = size
    hyperparameter_excel['intype'] = intype
    hyperparameter_excel['modelname'] = modelname
    hyperparameter_excel['schedular'] = schedular
    hyperparameter = pd.DataFrame(hyperparameter_excel,index=[0])
    # hyperparameter.to_excel(save_dir + '/hyperparameter_excel.xlsx', index=False)
    # 'D:/Datasets/'
    Dataset='.'
    train_covid_image_path = Dataset+size+'/train/COVID/images/'
    train_normal_image_path = Dataset+size+'/train/Normal/images/'
    train_pneumonia_image_path = Dataset+size+'/train/Viral Pneumonia/images/'
    train_lung_opacity_image_path = Dataset+size+'/train/Lung_Opacity/images/'
    # Paths to images
    val_covid_image_path = Dataset+size+test+'/COVID/images/'
    val_normal_image_path = Dataset+size+test+'/Normal/images/'
    val_pneumonia_image_path = Dataset+size+test+'/Viral Pneumonia/images/'
    val_lung_opacity_image_path = Dataset+size+test+'/Lung_Opacity/images/'

    # Paths to masks
    train_covid_mask_path = Dataset+size+'/train/COVID/masks/'
    train_normal_mask_path = Dataset+size+'/train/Normal/masks/'
    train_pneumonia_mask_path = Dataset+size+'/train/Viral Pneumonia/masks/'
    train_lung_opacity_mask_path = Dataset+size+'/train/Lung_Opacity/masks/'
    # Paths to masks
    val_covid_mask_path = Dataset+size+test+'/COVID/masks/'
    val_normal_mask_path = Dataset+size+test+'/Normal/masks/'
    val_pneumonia_mask_path = Dataset+size+test+'/Viral Pneumonia/masks/'
    val_lung_opacity_mask_path = Dataset+size+test+'/Lung_Opacity/masks/'

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
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=6, pin_memory=False)
    if modelname == '/Vit_Pretrain' or modelname == '/Vit':
        valset = NIH_Dataset(val_image_paths,  val_mask_paths, transform=valid_data_transform_vit, train=0, type=type)
    else:
        valset = NIH_Dataset(val_image_paths,  val_mask_paths, transform=valid_data_transform, train=0, type=type)
    validloader = DataLoader(valset, batch_size=2, shuffle=False, num_workers=4)
    if modelname == '/Proposed_Pretrain':
        model = our_model(num_classes=4)
        # pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\NIH_our_1.pkl')
        # net_state_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head') if k in net_state_dict}
        #net_state_dict.update(pretrained_dict)
        #model.load_state_dict(net_state_dict)
        pretrained_dict = torch.load('./resnet34-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc') if k in net_state_dict}

    if modelname == '/Resnet18_Pretrain':
        model = resnet18(num_classes=4)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet18-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc') if k in net_state_dict}
    if modelname == '/Res18-Vit-T_Pretrain':
        model = res18_vit_T(num_classes=4)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\NIH_resvit1.pkl')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head') if k in net_state_dict}
    if modelname == '/Efficient-Vit-T_Pretrain':
        model = efficient_vit_T(num_classes=4)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\NIH_effvit.pkl')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head') if k in net_state_dict}
    elif modelname == '/Resnet50_Pretrain':
        model = resnet50(num_classes=4)
        # 将模型写入tensorboard
        # init_img = torch.zeros((1, 3, 256, 256), device=device)
        # tb_writer.add_graph(model, init_img)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet50-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc') if k in net_state_dict}
    elif modelname == '/ResneXt50_Pretrain':
        model = resnext50_32x4d1(num_classes=4)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resneXt50-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc') if k in net_state_dict}
    elif modelname == '/Vit8':
        model = res18_vit8_T(num_classes=4)
    elif modelname == '/Resnet18':
        model = resnet18(num_classes=4)
    elif modelname == '/Resnet50':
        model = resnet50(num_classes=4)
    elif modelname == '/ResneXt50':
        model = resnext50_32x4d1(num_classes=4)
    elif modelname == '/Densenet121':
        model = densenet121(num_classes=4)
    elif modelname == '/Vit-16':
        model = vit_tiny_patch16_224(num_classes=4)
    elif modelname == '/Vit-32':
        model = vit_base_patch32(num_classes=4)
    elif modelname == '/ConvNeXt':
        model = convnext_tiny(num_classes=4)
    elif modelname == '/EfficientNetV2':
        model = efficientnetv2_s(num_classes=4)
    elif modelname == '/Res18-Vit-T':
        model = res18_vit_T(num_classes=4)
    elif modelname == '/Efficient-Vit-T':
        model = efficient_vit_T(num_classes=4)
    elif modelname == '/Proposed':
        model = our_model(num_classes=4)

    if Pre == 1:
        net_state_dict.update(pretrained_dict)
        model.load_state_dict(net_state_dict)
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=0.000001, weight_decay=5e-4)
    else:
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=0.00001, weight_decay=5e-4)
        #optimizer = optim.SGD(pg, lr=0.0001, momentum=0.9, weight_decay=5e-4)
    if lrs == 1:
        scheduler = create_lr_scheduler(optimizer, len(trainloader), epochs,
                                       warmup=True, warmup_epochs=10, end_factor=1e-5)
    criterion = nn.CrossEntropyLoss(weight=None)

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