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
from dataloader import NIH_Dataset, train_data_transform, valid_data_transform
from model_convnext import convnext_tiny
from model_resnet import resnet50, resnext50_32x4d1, resnet101, resnet18

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler

from utils import iou_coef, Hyperparameter, create_lr_scheduler


def main():

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        prediction = []
        true_labels = []

        for img, mask, y in tqdm(trainloader):
            images = img.type(torch.FloatTensor).to(device)
            masks = mask.type(torch.FloatTensor).to(device)
            y = y.type(torch.LongTensor).to(device)
            model.train()
            gamma1 = (images) ** 0.5
            complement1 = (1 - images)
            a = np.random.rand(224, 224)
            a = a > 0.5
            local_nose = torch.tensor(a.astype(int)) * (1 - masks) * images + masks * images
            p1 = local_nose[0].cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(p1[:, :, 0], cmap=plt.get_cmap('gray'))
            plt.show()
            aug_img1 = images
            gamma = 0.9832* (1 - masks) * images + (masks * images) ** 0.5
            complement = 1.0009 * (1 - masks) * images + ((1 - (masks) * images) * (masks))
            aug_img = 0.9946 * (1 - masks) * images + masks * images
            p1 = gamma[0].cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(p1[:, :, 0], cmap=plt.get_cmap('gray'))
            plt.show()
            p1 = complement[0].cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(p1[:, :, 0], cmap=plt.get_cmap('gray'))
            plt.show()
            p1 = aug_img[0].cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(p1[:, :, 0], cmap=plt.get_cmap('gray'))
            plt.show()
            cat = torch.cat([aug_img, aug_img1, gamma], dim=1)
            p1 = cat[0].cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(p1[:, :, :], cmap=plt.get_cmap('gray'))
            plt.show()
            print('cat',cat[0][2][55][100:112])
            img = torch.cat([images, images, images], dim=1)
            p1 = img[0].cpu().detach().numpy().transpose((1, 2, 0))
            plt.imshow(p1[:, :, :], cmap=plt.get_cmap('gray'))
            plt.show()
            print('img',img[0][1][55][100:112])

            out= model(images, masks)

    #         loss = criterion(out, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if lrs == 1:
    #             scheduler.step()
    #         acc = (out.argmax(dim=1) == y).float().mean()
    #         epoch_accuracy += acc / len(trainloader)
    #         epoch_loss += loss / len(trainloader)
    #     tacc.append(epoch_accuracy.cpu().detach().numpy())
    #     tloss.append(epoch_loss.cpu().detach().numpy())
    #
    #
    #     with torch.no_grad():
    #         prediction = []
    #         true_labels = []
    #         epoch_val_accuracy = 0
    #         epoch_val_loss = 0
    #         # 计算验证集准确率
    #         imgs = []
    #         for img, mask, y in tqdm(validloader):
    #             images = img.type(torch.FloatTensor).to(device)
    #             masks = mask.type(torch.FloatTensor).to(device)
    #             vy = y.type(torch.LongTensor).to(device)
    #
    #             model.eval()
    #             val_output = model(images, masks)
    #             val_loss = criterion(val_output, vy)
    #
    #             imgs.append(images)
    #             acc = (val_output.argmax(dim=1) == vy).float().mean()
    #             epoch_val_accuracy += acc / len(validloader)
    #             epoch_val_loss += val_loss / len(validloader)
    #             prediction += val_output.argmax(dim=1).tolist()
    #             true_labels += vy.tolist()
    #         p1 = imgs[0][0].cpu().detach().numpy().transpose((1, 2, 0))
    #
    #         vloss.append(epoch_val_loss.cpu().detach().numpy())
    #         vacc.append(epoch_val_accuracy.cpu().detach().numpy())
    #         if min(vloss1) > epoch_val_loss:
    #             estop = 0
    #             print("最优epoch: ", epoch + 1)
    #             best = epoch+1
    #             # best_acc = epoch_val_accuracy
    #             best_loss = epoch_val_loss
    #             torch.save(model.state_dict(), save_dir+'/parameters.pkl')
    #             print("a: ", model.state_dict()['a'])
    #             print("b: ", model.state_dict()['b'])
    #             print("c: ", model.state_dict()['c'])
    #             print("r: ", model.state_dict()['r'])
    #             print("k: ", model.state_dict()['k'])
    #         else:
    #             estop += 1
    #             if lrs == 0:
    #                 print("Counter {} of 12".format(estop))
    #                 if estop > 12:
    #                     print("Early stopping with best_acc: ", best_loss, "and val_acc for this epoch: ", epoch_val_loss, "...")
    #                     break
    #             elif lrs == 1:
    #                 print("Counter {} of 22".format(estop))
    #                 if estop > 22:
    #                     print("Early stopping with best_acc: ", best_loss, "and val_acc for this epoch: ", epoch_val_loss, "...")
    #                     break
    #         vloss1.append(epoch_val_loss)
    #         tags = ["loss", "accuracy", "learning_rate"]
    #         tb_writer.add_scalars(tags[0], {'train_loss': epoch_loss.cpu().detach().numpy(),
    #                                         'val_loss': epoch_val_loss.cpu().detach().numpy()}, epoch)
    #         tb_writer.add_scalars(tags[1], {'train_acc': epoch_accuracy.cpu().detach().numpy(),
    #                                         'val_acc': epoch_val_accuracy.cpu().detach().numpy()}, epoch)
    #         tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
    #         # add conv1 weights into tensorboard
    #         # tb_writer.add_histogram(tag="conv1",
    #         #                         values=model.conv1.weight,
    #         #                         global_step=epoch)
    #         # tb_writer.add_histogram(tag="layer4/block0/conv1",
    #         #                         values=model.layer4[0].conv1.weight,
    #         #                         global_step=epoch)
    #     confusion_mtx = confusion_matrix(true_labels, prediction)
    #
    #     lr = optimizer.state_dict()['param_groups'][0]['lr']
    #     print(
    #         f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - lr：{lr:.7f}\n"
    #         )
    #     print(confusion_mtx)
    #     print(best, best_loss)
    #
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
    tb_writer = SummaryWriter(log_dir="runs/chest_experiment")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 超参数
    lrs = 1
    Pre = 0
    size, intype, modelname, schedular, epochs = Hyperparameter(size=2, Pre=Pre, model=0, lrs=lrs, type=3)
    save_dir = 'D:/ExpResult/train/' + size + intype + modelname + schedular
    hyperparameter_excel = {'data_size': [], 'intype': [], 'modelname': [], 'schedular': []}
    hyperparameter_excel['data_size'] = size
    hyperparameter_excel['intype'] = intype
    hyperparameter_excel['modelname'] = modelname
    hyperparameter_excel['schedular'] = schedular
    hyperparameter = pd.DataFrame(hyperparameter_excel,index=[0])
    hyperparameter.to_excel(save_dir + '/hyperparameter_excel.xlsx', index=False)
    train_covid_image_path = 'D:/Datasets/'+size+'/train/COVID/images/'
    train_normal_image_path = 'D:/Datasets/'+size+'/train/Normal/images/'
    train_pneumonia_image_path = 'D:/Datasets/'+size+'/train/Viral Pneumonia/images/'
    train_lung_opacity_image_path = 'D:/Datasets/'+size+'/train/Lung_Opacity/images/'
    # Paths to images
    val_covid_image_path = 'D:/Datasets/'+size+'/val/COVID/images/'
    val_normal_image_path = 'D:/Datasets/'+size+'/val/Normal/images/'
    val_pneumonia_image_path = 'D:/Datasets/'+size+'/val/Viral Pneumonia/images/'
    val_lung_opacity_image_path = 'D:/Datasets/'+size+'/val/Lung_Opacity/images/'

    # Paths to masks
    train_covid_mask_path = 'D:/Datasets/'+size+'/train/COVID/masks/'
    train_normal_mask_path = 'D:/Datasets/'+size+'/train/Normal/masks/'
    train_pneumonia_mask_path = 'D:/Datasets/'+size+'/train/Viral Pneumonia/masks/'
    train_lung_opacity_mask_path = 'D:/Datasets/'+size+'/train/Lung_Opacity/masks/'
    # Paths to masks
    val_covid_mask_path = 'D:/Datasets/'+size+'/val/COVID/masks/'
    val_normal_mask_path = 'D:/Datasets/'+size+'/val/Normal/masks/'
    val_pneumonia_mask_path = 'D:/Datasets/'+size+'/val/Viral Pneumonia/masks/'
    val_lung_opacity_mask_path = 'D:/Datasets/'+size+'/val/Lung_Opacity/masks/'

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

    trainset = NIH_Dataset(train_image_paths, train_mask_paths, transform=train_data_transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valset = NIH_Dataset(val_image_paths,  val_mask_paths, transform=valid_data_transform_vit)
    validloader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=2)
    if modelname == '/Resnet50_Pretrain':
        model = resnet50(num_classes=4)
        # 将模型写入tensorboard
        # init_img = torch.zeros((1, 3, 256, 256), device=device)
        # tb_writer.add_graph(model, init_img)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\resnet50-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc') if k in net_state_dict}

    elif modelname == '/ConvNeXt_Pretrain':
        model = convnext_tiny(num_classes=4)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\convnext_tiny-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head') if k in net_state_dict}
    elif modelname == '/Vit_Pretrain':
        model = vit_base_patch16_224(num_classes=4)
        pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\vit_base-imagenet.pth')
        net_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head') if k in net_state_dict}
    elif modelname == '/Resnet18':
        model = resnet18(num_classes=4)
    elif modelname == '/Resnet50':
        model = resnet50(num_classes=4)
    elif modelname == '/ConvNeXt':
        model = convnext_tiny(num_classes=4)
    elif modelname == '/Vit':
        model = vit_base_patch16_224(num_classes=4)
    if Pre == 1:
        net_state_dict.update(pretrained_dict)
        model.load_state_dict(net_state_dict)

    # 新的优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.0001, weight_decay=5e-3)# resnet-lr：0.0001 convnext-lr：0.0005 vit-lr：0.0001
    #optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-3)
    if lrs == 1:
        scheduler = create_lr_scheduler(optimizer, len(trainloader), epochs,
                                       warmup=True, warmup_epochs=6)
    weight1 = torch.from_numpy(np.array([1.0, 1.27, 0.743, 1.06])).float()
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