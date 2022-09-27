# -- coding: utf-8 --
import torch
from torchsummary import summary
from torchvision import models

from D_C.mode_our_1_c import our_model_1_c
from D_C.model_densenet_c import densenet121_c

from mode_our_1 import our_model
from mode_our_2 import our_model_2
from mode_res18_vit import res18_vit_T
from mode_vit import vit_tiny_patch16_224, vit_base_patch32
from model_convnext import convnext_tiny
from model_densenet import densenet121
from model_efficientnetV2 import efficientnetv2_s
from model_resnet import resnet18, resnet50, resnext50_32x4d1

#summary(resnet18(num_classes=4).cuda(), (3, 256, 256))
# summary(resnet50(num_classes=4).cuda(), (3, 256, 256))
# summary(densenet121(num_classes=4).cuda(), (3, 256, 256))
# summary(resnext50_32x4d1(num_classes=4).cuda(), (3, 256, 256))
# summary(convnext_tiny(num_classes=4).cuda(), (3, 256, 256))
# summary(efficientnetv2_s(num_classes=4).cuda(), (3, 256, 256))
#summary(vit_tiny_patch16_224(num_classes=4).cuda(), (3, 256, 256))
# summary(vit_base_patch32(num_classes=4).cuda(), (3, 256, 256))
# summary(res18_vit_T(num_classes=4).cuda(), (3, 256, 256))

summary(our_model_2(num_classes=4).cuda(), (3, 256, 256))
#summary(our_model(num_classes=4).cuda(), (3, 256, 256))
# summary(our_model_1_c(num_classes=4).cuda(), (3, 256, 256))
# summary(densenet121_c(num_classes=4).cuda(), (3, 256, 256))
# print(densenet121(num_classes=4))
