import torch
from PIL import Image
from pylab import *
from torchvision import datasets, models, transforms as T
from torch.utils.data import Dataset, DataLoader
#import albumentations as A
#from torchtoolbox.transform import Cutout
#import tensorflow.keras.backend as K
import torch.nn.functional as F



class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# 直方图均衡化
class histeq (object):
    def __init__(self, x=0):
        self. x = x

    def __call__(self, im, nbr_bins=256):
        imhist, bins = histogram(im.flatten())
        cdf = imhist.cumsum()
        cdf = cdf / cdf[-1]
        # 使用累积分布函数的线性插值，计算新的像素值
        im2 = interp(im.flatten() ,bins[:-1] ,cdf)
        return im2.reshape(im.shape)

def iou_coef(y_true, y_pred, smooth=1):
    y_pred=torch.sigmoid(y_pred)
    intersection = torch.sum(torch.abs(y_true * y_pred), axis=[1,2,3])
    union = torch.sum(y_true,[1,2,3])+torch.sum(y_pred,[1,2,3])-intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# 数据增强方式
def data_transforms(phase=None):
    if phase == 'train':
        data_T = T.Compose([
            T.Resize(size=(256, 256)),
            #AddPepperNoise(0.9, p=0.5),
            #Cutout(p=0.5, scale=(0.01, 0.01)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=(-20, +20)),
            #T.CenterCrop(size=256),
            #T.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254 / 255, 0, 127 / 255)),
            T.RandomAffine(translate=(0.20, 0.20), degrees=0),
            #T.RandomGrayscale(),
            T.ToTensor(),
            #A.Cutout(num_holes=45, max_h_size=15, max_w_size=15, fill_value=0, always_apply=False, p=1),
            #T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            #histeq(),
        ])
    elif phase == 'test' or phase == 'val':
        data_T = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            # T.Grayscale(num_output_channels=1),
            #T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            #histeq(),
        ])
    return data_T


def Hyperparameter(size=0, Pre=0, model=0, lrs=0, type=0):
    if size == 0:
        datasize = "tiny"
    elif size == 1:
        datasize = "small"
    elif size == 2:
        datasize = "/base"
    elif size == 3:
        datasize = "chest_3class2"


    if type == 0:
        intype = '/0img'
    elif type == 1:
        intype = '/1aug_img'
    elif type == 2:
        intype = '/2mask'
    elif type == 3:
        intype = '/3aug_mask'
    elif type == 4:
        intype = '/4aug1'
    elif type == 5:
        intype = '/5aug2'
    elif type == 6:
        intype = '/6channel1'
    elif type == 7:
        intype = '/7channel2'


    if model == 0:
        modelname = "/Resnet18"
    elif model == 1:
        modelname = "/Resnet50"
    elif model == 2:
        modelname = "/Densenet121"
    elif model == 3:
        modelname = "/ResneXt50"
    elif model == 4:
        modelname = "/Vit-16"
    elif model == 5:
        modelname = "/ConvNeXt"
    elif model == 6:
        modelname = "/EfficientNetV2"
    elif model == 7:
        modelname = "/Vit-32"
    elif model == 8:
        modelname = "/Res18-Vit-T"
    elif model == 9:
        modelname = "/Efficient-Vit-T"
    elif model == 10:
        modelname = "/Proposed"
    elif model == 11:
        modelname = "/Resnet18_c"
    elif model == 12:
        modelname = "/Densenet121_c"
    elif model == 13:
        modelname = "/EfficientNetV2_c"
    elif model == 14:
        modelname = "/Res18-Vit-T_c"
    elif model == 15:
        modelname = "/Proposed_c"
    elif model == 16:
        modelname = "/Vit8"
    elif model == 17:
        modelname = "/tttt"





    if lrs == 1:
        schedular = '/scheduler'
    elif lrs == 0:
        schedular = '/noscheduler'
    if Pre == 1:
        modelname = modelname + '_Pretrain'
        if size == 0:
            epochs = 20
        if size == 1:
            epochs = 40
        if size == 2:
            epochs = 200
        if size == 3:
            epochs = 200
        if size == 4:
            epochs = 200

    else:

        if size == 0:
            epochs = 100
        if size == 1:
            epochs = 150
        if size == 2:
            epochs = 200
        if size == 3:
            epochs = 200


    return datasize, intype, modelname, schedular, epochs


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_epochs1=1,
                        warmup_factor=2e-1,
                        end_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            #return warmup_factor * (1 - alpha) + alpha
            return 1
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs1) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return (((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (0.5 - end_factor) + end_factor)*1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



