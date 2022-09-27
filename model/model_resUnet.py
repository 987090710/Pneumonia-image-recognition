import math
import random

import torch.nn as nn
import torch

from model_convnext import LayerNorm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out + identity
        out = self.relu(out)

        return out


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2, bilinear=True):
        super(up_conv, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=kernel_size,
                                            stride=strides, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    def forward(self, x):
        if self.bilinear:
            out = self.up(x)
            out = self.conv1(out)
            out = self.bn(out)
            out = self.act(out)
            out = self.conv2(out)
        else:
            out = self.act((self.conv1(x)))
        return out


class pcam(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pcam, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.act1 = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        logit = self.conv1(x)
        map = self.act1(logit)
        map = self.bn(map)
        out = x*map
        out = self.avgpool(out)
        out = self.conv2(out)
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # self.up1 = up_conv(256, 128)
        # self.up2 = up_conv(128, 64)
        # self.up3 = up_conv(64, 1)
        # self.up4 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.pcam1 = pcam(256, 256)
        self.pcam2 = pcam(256, 256)
        self.pcam3 = pcam(256, 256)
        self.pcam4 = pcam(256, 256)
        self.pcam5 = pcam(256, 256)
        self.pcam6 = pcam(256, 256)
        self.pcam7 = pcam(256, 256)
        self.pcam8 = pcam(256, 256)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.include_top:
            out1 = self.pcam1(x)
            out2 = self.pcam2(x)
            out3 = self.pcam3(x)
            out4 = self.pcam4(x)
            # out5 = self.pcam5(x)
            # out6 = self.pcam6(x)
            # out7 = self.pcam7(x)
            # out8 = self.pcam8(x)
            # , out5, out6, out7, out8
            out = torch.cat([out1, out2, out3, out4], dim=1)
            out = torch.flatten(out, 1)

        return out


def resUnet(num_classes=4, include_top=True):

    return ResNet(BasicBlock, [2, 6, 3], num_classes=num_classes, include_top=include_top)

