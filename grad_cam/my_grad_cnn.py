import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms

from dataloader import valid_data_transform, NIH_Dataset
from model_resnet import resnet50
from utils import GradCAM, show_cam_on_image, center_crop_img, Hyperparameter
import torchvision.transforms as T

def main():

    model = resnet50(num_classes=4)
    pretrained_dict = torch.load(r'D:\ExpResult\pretrain_parameters\best50_pretrain.pkl')
    net_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict)
    model.load_state_dict(net_state_dict)
    test = "/val"
    size = "/base"
    test_covid_image_path = 'D:/Datasets/' + size + test + '/COVID/images/'
    test_normal_image_path = 'D:/Datasets/' + size + test + '/Normal/images/'
    test_pneumonia_image_path = 'D:/Datasets/' + size + test + '/Viral Pneumonia/images/'
    test_lung_opacity_image_path = 'D:/Datasets/' + size + test + '/Lung_Opacity/images/'
    test_covid_mask_path = 'D:/Datasets/' + size + test + '/COVID/masks/'
    test_normal_mask_path = 'D:/Datasets/' + size + test + '/Normal/masks/'
    test_pneumonia_mask_path = 'D:/Datasets/' + size + test + '/Viral Pneumonia/masks/'
    test_lung_opacity_mask_path = 'D:/Datasets/' + size + test + '/Lung_Opacity/masks/'
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

    img,mask,label = NIH_Dataset(test_image_paths, test_mask_paths, transform=valid_data_transform, train=0, type=type).__getitem__(0)








    target_layers = [model.layer4]
    resize = T.Resize(size=(256, 256))
    totensor = T.ToTensor()
    topil = T.ToPILImage()

    # load image
    img_path = "D:/ExpResult/gradcam/pic/COVID-900.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('L')
    # img = resize(img)
    # img = totensor(img)
    # img = torch.cat([img, img, img], dim=0)
    # img = topil(img)
    img0 = np.array(img, dtype=np.uint8).transpose((1, 2, 0))
    plt.imshow(img0)
    plt.show()
    # img = center_crop_img(img, 224)

    # [C, H, W]
    #img_tensor = totensor(img)


    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img, dim=0)

    out = model(input_tensor)
    print(out)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = out.argmax(dim=1)  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img0.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()
    plt.imsave("D:/ExpResult/gradcam/out/900.png", visualization)


if __name__ == '__main__':
    main()
