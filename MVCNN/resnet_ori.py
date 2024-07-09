import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from .triplet_attention import *


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
model_triplet_url = '/root/data1/code/PES-main/RESNET50_PCAM_IMAGENET_model_best.pth.tar'

# 0是triplet_attention 1是SeNet  2表示CBAM 3表示BAM
att_list = {"TripletAttention" : 0, "SENet" : 1, "CBAM": 2, "BAM": 3} 

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, use_triplet_attention=False
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_triplet_attention:
            self.triplet_attention = TripletAttention(planes, 16)
        else:
            self.triplet_attention = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.triplet_attention is None:
            out = self.triplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, use_triplet_attention=False
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_triplet_attention:
            self.triplet_attention = TripletAttention(planes * 4, 16)
        else:
            self.triplet_attention = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.triplet_attention is None:
            out = self.triplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None, triplet_pos = 0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if network_type == "ImageNet":
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 只在最后一层加triplet
        if triplet_pos == 2:
            self.layer1 = self._make_layer(block, 64, layers[0], triplet_pos = triplet_pos)
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=2,  triplet_pos = triplet_pos
            )
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2, triplet_pos = triplet_pos
            )
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2, att_type=att_type, triplet_pos = triplet_pos
            )
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type,triplet_pos = triplet_pos)
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=2, att_type=att_type, triplet_pos = triplet_pos
            )
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2, att_type=att_type, triplet_pos = triplet_pos
            )
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2, att_type=att_type, triplet_pos = triplet_pos
            )
        if network_type == "ImageNet":
            self.avgpool = nn.AvgPool2d(7)        

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode="fan_out")
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, triplet_pos = 0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []        
        # 所有的block全都加上
        if triplet_pos == 0 or triplet_pos == 2:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample,
                    use_triplet_attention= att_type == "TripletAttention",
                )
            )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        use_triplet_attention=att_type == "TripletAttention",
                    )
                )
        # 每一层的最后的block加上
        elif triplet_pos == 1:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample
                )
            )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks - 1):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                    )
                )
            layers.append(
                    block(
                        self.inplanes,
                        planes,
                        use_triplet_attention=att_type == "TripletAttention",
                    )
                )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)
        # print("before layer1,shape = ",x.shape)
        x = self.layer1(x)
        # print("after layer1,shape = ",x.shape)
        x = self.layer2(x)
        # print("after layer2,shape = ",x.shape)
        x = self.layer3(x)
        # print("after layer3,shape = ",x.shape)
        x = self.layer4(x)
        # print("after layer4,shape = ",x.shape)
        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        # print("resnet2222")
        x = x.view(x.size(0), -1)
        # print("before fc,shape = ",x.shape)
        out = self.fc(x)
        return x, out #特征 & 分类结果


def ResidualNet(network_type, depth, num_classes, att_type=None, pretrain=True, triplet_pos = 0, pretrain_add = 0):

    assert network_type in [
        "ImageNet",
        "CIFAR10",
        "CIFAR100",
    ], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], "network depth should be 18, 34, 50 or 101"

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, triplet_pos)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type, triplet_pos)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type, triplet_pos)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type, triplet_pos)

    #加载预训练模型
    arch_name = 'resnet' + str(depth)
    if pretrain:
        if att_type == None or pretrain_add == 1:
            state_dict = load_state_dict_from_url(model_urls[arch_name], progress=False)
            # for k, v in state_dict.items():
            #     print(k)
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            # print("--------------")
            # for name, _ in model.named_parameters():
            #     print(name)
            model.load_state_dict(state_dict, strict=False)# ,strict=False
            print("=> loaded checkpoint '{}'".format(model_urls[arch_name]))
        elif att_type == "TripletAttention":
            # triplet attention 官方预训练pthtar
            # checkpoint = torch.load(model_triplet_url, map_location={'cuda:0':'cuda:1'})
            checkpoint = torch.load(model_triplet_url)
            state_dict = checkpoint['state_dict']
            # model.load_state_dict(state_dict)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v

                new = k[7:].split('.')
                if len(new)>2 and new[2]=='pcam':
                    del new_state_dict[k[7:]]
                    new[2] = 'triplet_attention'
                    new_name = ''
                    for i in new:
                        new_name = new_name + i + '.'
                    new_state_dict[new_name[:-1]] = v
                    # print(new_name[:-1])
            del new_state_dict['fc.bias']
            del new_state_dict['fc.weight']
            model.load_state_dict(new_state_dict, strict=False) 
            # model.load_state_dict(new_state_dict)                 
            # if "optimizer" in checkpoint:
            #     optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint '{}'".format(model_triplet_url))
    return model
