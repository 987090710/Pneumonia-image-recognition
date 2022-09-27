import os
import math
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim, nn
from Unet import UNet

from Loader import NIH_Dataset, valid_data_transform
from grad_cam.utils import GradCAM, show_cam_on_image


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

    i = 0
    for data, class_name,image_name in tqdm(testloader):
        i = i + 1
        data = data.type(torch.FloatTensor)
        data = data.to(device)

        model.eval()
        test_output = model(data)

        mask = test_output[0].cpu().detach().numpy().transpose((1, 2, 0))
        if not os.path.exists(save_dir +str(class_name[0])+"/masks/"):
            os.makedirs(save_dir +str(class_name[0])+"/masks/")

        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        # 设置子图占满整个画布
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # 关掉x和y轴的显示
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(mask, cmap='viridis')
        plt.savefig(save_dir +str(class_name[0])+"/masks/"+str(image_name[0]))
        #plt.show()





if __name__ == '__main__':
    lrs = 1
    Pre = 0
    type = 0
    size, intype, modelname, schedular, epochs = Hyperparameter(size=2, Pre=Pre, model=2, lrs=lrs, type=type)
    save_dir = 'D:/ExpResult/out_segment/'
    residual_test = 0
    test = ""
    size = "NIH8"
    test_Atelectasis_image_path = 'D:/Datasets/'+size+test+'/Atelectasis/images/'
    test_Cardiomegaly_image_path = 'D:/Datasets/'+size+test+'/Cardiomegaly/images/'
    test_Effusion_image_path = 'D:/Datasets/'+size+test+'/Effusion/images/'
    test_Infiltrate_image_path = 'D:/Datasets/'+size+test+'/Infiltrate/images/'
    test_Mass_image_path = 'D:/Datasets/' + size + test + '/Mass/images/'
    test_Nodule_image_path = 'D:/Datasets/' + size + test + '/Nodule/images/'
    test_Pneumonia_image_path = 'D:/Datasets/' + size + test + '/Pneumonia/images/'
    test_Pneumothorax_image_path = 'D:/Datasets/' + size + test + '/Pneumothorax/images/'

    test_image_paths = [[test_Atelectasis_image_path + file for file in os.listdir(test_Atelectasis_image_path)]
                       + [test_Cardiomegaly_image_path + file for file in os.listdir(test_Cardiomegaly_image_path)]
                       + [test_Effusion_image_path + file for file in os.listdir(test_Effusion_image_path)]
                       + [test_Infiltrate_image_path + file for file in os.listdir(test_Infiltrate_image_path)]
                       + [test_Mass_image_path + file for file in os.listdir(test_Mass_image_path)]
                       + [test_Nodule_image_path + file for file in os.listdir(test_Nodule_image_path)]
                       + [test_Pneumonia_image_path + file for file in os.listdir(test_Pneumonia_image_path)]
                       + [test_Pneumothorax_image_path + file for file in os.listdir(test_Pneumothorax_image_path)]
                       ][0]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = NIH_Dataset(test_image_paths, transform=valid_data_transform, train=0, type=type)
    validloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

    model = UNet()
    pretrained_dict = torch.load(r'D:\ExpResult\train_segment\run20.pkl')
    net_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict)
    model.load_state_dict(net_state_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    vacc = [0]
    vloss = []
    main()