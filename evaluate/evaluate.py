import os
import math
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim, nn



from dataloader import NIH_Dataset, valid_data_transform, valid_data_transform_vit
from grad_cam.utils import GradCAM, show_cam_on_image
from mode_efficient_vit import efficient_vit_T
from mode_our_1 import our_model
from mode_res18_vit import res18_vit_T
from mode_resvit import resvit
from mode_vit import vit_tiny_patch16_224
from model_convnext import convnext_tiny
from model_densenet import densenet121
from model_efficientnetV2 import efficientnetv2_s
from model_resnet import resnet50, resnext50_32x4d1, resnet18, resnet34

from utils import data_transforms, iou_coef, Hyperparameter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import metrics
import seaborn as sns

def main():
    #with torch.no_grad():
    testloader = DataLoader(testset, batch_size=1, num_workers=4)
    # 模型在测试集上的表现

    test_accuracy = 0.
    predictions = []
    true_labels = []
    i = 0
    for data, mask, label in tqdm(testloader):
        i = i + 1
        data = data.type(torch.FloatTensor)
        data = data.to(device)

        label = label.to(device)
        model.eval()
        test_output = model(data)
        # p = p[0].cpu().detach().numpy().transpose((1, 2, 0))
        # plt.imshow(p[1])
        # plt.show()
        predictions += test_output.argmax(dim=1).tolist()
        true_labels += label.tolist()

        acc = (test_output.argmax(dim=1) == label).float().mean()
        test_accuracy += acc / len(testloader)


        #target_layers = [model.layer4]

        target_layers = [model.denseblock4.denselayer16]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = test_output.argmax(dim=1)  # tabby, tabby cat
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        img = data[0].cpu().detach().numpy().transpose((1, 2, 0))
        mask0 = mask[0].cpu().detach().numpy().transpose((1, 2, 0))
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32),
                                          grayscale_cam,
                                          mask0=mask0,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.imsave(save_dir+"/cam1/" + str(i) + ".png", visualization)

    print('test_accuracy', test_accuracy.cpu().detach().numpy())
    print(metrics.classification_report(true_labels, predictions))

    confusion_mtx = confusion_matrix(true_labels, predictions)
    confusion_mtx1 = np.array(confusion_mtx)
    # sum = [365, 620, 1001, 131]
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
    Pre = 0
    type = 2
    size, intype, modelname, schedular, epochs = Hyperparameter(size=2, Pre=Pre, model=2, lrs=lrs, type=type)
    save_dir = 'D:/ExpResult/train/' + size + intype + modelname + schedular
    os.makedirs(save_dir + "/cam1")
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
    if modelname == '/Resnet18_Pretrain':
        model = resnet18(num_classes=4)
    elif modelname == '/Resnet50_Pretrain':
        model = resnet50(num_classes=4)
    elif modelname == '/Resnet18':
        model = resnet18(num_classes=4)
    elif modelname == '/ResneXt50':
        model = resnext50_32x4d1(num_classes=4)
    elif modelname == '/Resnet50':
        model = resnet50(num_classes=4)
    elif modelname == '/Vit_Pretrain':
        model = vit_tiny_patch16_224(num_classes=4)
    elif modelname == '/Vit':
        model = vit_tiny_patch16_224(num_classes=4)
    elif modelname == '/ConvNeXt':
        model = convnext_tiny(num_classes=4)
    elif modelname == '/Densenet121':
        model = densenet121(num_classes=4)
    elif modelname == '/EfficientNetV2':
        model = efficientnetv2_s(num_classes=4)
    elif modelname == '/Res18-Vit-T' or modelname == '/Res18-Vit-T_Pretrain':
        model = res18_vit_T(num_classes=4)
    elif modelname == '/Efficient-Vit-T' or modelname == '/Efficient-Vit-T_Pretrain':
        model = efficient_vit_T(num_classes=4)
    elif modelname == '/Proposed' or modelname == '/Proposed_Pretrain':
        model = our_model(num_classes=4)


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