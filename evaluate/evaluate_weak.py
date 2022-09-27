import os
import math
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim, nn

from D_C.mode_our_1_c import our_model_1_c
from D_C.mode_our_c import our_model_c
from D_C.mode_res18_vit_c import res18_vit_T_c
from D_C.model_densenet_c import densenet121_c
from D_C.model_efficientnetV2_c import efficientnetv2_s_c
from D_C.model_resnet_c import resnet18_c
from dataloader import NIH_Dataset, valid_data_transform, valid_data_transform_vit
from grad_cam.utils import GradCAM, show_cam_on_image
from mode_resvit1 import resvit, resvit_base, resvit_large
from model_convnext import convnext_tiny
from model_resnet import resnet50, resnext50_32x4d1, resnet18
import torchvision.transforms as T

from utils import data_transforms, iou_coef, Hyperparameter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import metrics
import seaborn as sns


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    # with torch.no_grad():
    testloader = DataLoader(testset, batch_size=1, num_workers=4)
    # 模型在测试集上的表现
    resize = T.Resize(size=(256, 256))
    test_accuracy = 0.
    predictions = []
    true_labels = []
    i=0
    for data, mask, label in tqdm(testloader):
        i=i+1
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        mask = mask.type(torch.FloatTensor)
        label = label.to(device)
        model.eval()
        test_output, out = model(data)
        # for i in range(1):
        #     p = resize(out[0]).cpu().detach().numpy().transpose((1, 2, 0))
        #     # 设置子图占满整个画布
        #     fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     # 关掉x和y轴的显示
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     ax.imshow(p[:,:,i], cmap='gray')
        # #     # plt.show()
        #     plt.savefig(save_dir + "/feature/" + f"{i}" + '.png')
        #     p1 = xx[0].cpu().detach().numpy().transpose((1, 2, 0))
        #     plt.imshow(p1[:,:,i])
        #     plt.show()
        # p = mask[0].cpu().detach().numpy().transpose((1, 2, 0))
        # plt.imshow(p)
        # plt.show()
        predictions += test_output.argmax(dim=1).tolist()
        true_labels += label.tolist()

        acc = (test_output.argmax(dim=1) == label).float().mean()
        test_accuracy += acc / len(testloader)

        # target_layers = [ model.blocks[0].norm1, model.blocks[-1].norm1]
        # cam = GradCAM(model=model,
        #               target_layers=target_layers,
        #               use_cuda=False,
        #               reshape_transform=ReshapeTransform(model))
        # target_category = test_output.argmax(dim=1)  # tabby, tabby cat
        # img = data[0].cpu().detach().numpy().transpose((1, 2, 0))
        # mask0 = resize(out[0]).cpu().detach().numpy().transpose((1, 2, 0))
        # mask1 = mask[0].cpu().detach().numpy().transpose((1, 2, 0))
        # mask0=(mask0+mask1)/2
        # grayscale_cam = cam(input_tensor=data, target_category=target_category)
        # grayscale_cam = grayscale_cam[0, :]
        # visualization = show_cam_on_image(img, grayscale_cam, mask0, use_rgb=True)
        # plt.imshow(visualization)
        # plt.imsave(save_dir+"/cam1/" + str(i) + ".png", visualization)
        target_layers = [model.layer4]

        # target_layers = [model.blocks2[29]]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = test_output.argmax(dim=1)  # tabby, tabby cat
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        img = data[0].cpu().detach().numpy().transpose((1, 2, 0))
        mask0 = resize(out[0]).cpu().detach().numpy().transpose((1, 2, 0))
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32),
                                          grayscale_cam,
                                          mask0=mask0,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.imsave(save_dir + "/cam1/" + str(i) + ".png", visualization)


    print('test_accuracy', test_accuracy.cpu().detach().numpy())
    print(metrics.classification_report(true_labels, predictions))

    confusion_mtx = confusion_matrix(true_labels, predictions)
    confusion_mtx1 = np.array(confusion_mtx)
    #sum = [364, 619, 997, 136]
    # sum = [250, 250, 250, 250]
    # for i in range(4):
    #     for j in range(4):
    #         confusion_mtx1[i][j] = confusion_mtx[i][j]/sum[i]*100

    # plot the confusion matrix

    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="YlGnBu", linecolor="gray", fmt='.2f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion "+test+f": {test_accuracy:.4f}")
    myfig = plt.gcf()
    myfig.savefig(save_dir + "/best"+f"{test_accuracy:.4f}"+'.png')
    plt.show()


if __name__ == '__main__':
    lrs = 1
    Pre = 1
    type = 0
    size, intype, modelname, schedular, epochs = Hyperparameter(size=2, Pre=Pre, model=14, lrs=lrs, type=type)
    save_dir = 'D:/ExpResult/train/' + size + intype + modelname + schedular
    os.makedirs(save_dir+"/cam1")
    residual_test = 0
    if residual_test == 0:
        test = "/test"
    elif residual_test == 1:
        test = "/res_test"
    size = "grad_cam"
    test_covid_image_path = 'D:/Datasets/'+size+test+'/COVID/images/'
    test_normal_image_path = 'D:/Datasets/'+size+test+'/Normal/images/'
    test_pneumonia_image_path = 'D:/Datasets/'+size+test+'/Viral Pneumonia/images/'
    test_lung_opacity_image_path = 'D:/Datasets/'+size+test+'/Lung_Opacity/images/'
    test_covid_mask_path = 'D:/Datasets/'+size+test+'/COVID/masks/'
    test_normal_mask_path = 'D:/Datasets/'+size+test+'/Normal/masks/'
    test_pneumonia_mask_path = 'D:/Datasets/'+size+test+'/Viral Pneumonia/masks/'
    test_lung_opacity_mask_path = 'D:/Datasets/'+size+test+'/Lung_Opacity/masks/'
    test_image_paths = [[test_covid_image_path + file for file in os.listdir(test_covid_image_path)]
                       + [test_normal_image_path + file for file in os.listdir(test_normal_image_path)]
                       + [test_pneumonia_image_path + file for file in os.listdir(test_pneumonia_image_path)]
                       + [test_lung_opacity_image_path + file for file in os.listdir(test_lung_opacity_image_path)]
                       ][0]
    test_mask_paths = [[test_covid_mask_path + file for file in os.listdir(test_covid_mask_path)]
                      + [test_normal_mask_path + file for file in os.listdir(test_normal_mask_path)]
                      + [test_pneumonia_mask_path + file for file in os.listdir(test_pneumonia_mask_path)]
                      + [test_lung_opacity_mask_path + file for file in os.listdir(test_lung_opacity_mask_path)]
                      ][0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = NIH_Dataset(test_image_paths, test_mask_paths, transform=valid_data_transform_vit, train=0, type=type)
    validloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    if modelname == '/Proposed_c_Pretrain':
        model = our_model_1_c(num_classes=4)
    if modelname == '/Resvit_Pretrain':
        model = resvit(num_classes=4)
    elif modelname == '/Resvit':
        model = resvit(num_classes=4)
    elif modelname == '/Resvit_base_Pretrain':
        model = resvit_base(num_classes=4)
    elif modelname == '/Resnet18_c':
        model = resnet18_c(num_classes=4)
    elif modelname == '/Densenet121_c':
        model = densenet121_c(num_classes=4)
    elif modelname == '/EfficientNetV2_c':
        model = efficientnetv2_s_c(num_classes=4)
    elif modelname == '/Res18-Vit-T_c_Pretrain':
        model = res18_vit_T_c(num_classes=4)
    # 预训练
    pretrained_dict = torch.load(save_dir+'/best.pkl')
    net_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict)
    model.load_state_dict(net_state_dict)
    if torch.cuda.is_available():
        model = model.cuda()


    vacc = [0]
    vloss = []
    main()