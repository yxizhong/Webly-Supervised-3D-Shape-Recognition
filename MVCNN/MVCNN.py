import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
from .resnet import *


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=55, pretraining=True, cnn_name='vgg11', attention = None, triplet_pos = 0, pretrain_add = 0):
        super(SVCNN, self).__init__(name)
        # 15class
        # self.classnames=['airplane','bench','bookshelf','bowl','chair','cone','door','guitar','keyboard','laptop','person','sofa','stairs','stool','xbox']
        #40class
        # self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        #50class
        # self.classnames = ['airplane&aeroplane&plane', 'ashcan&trash_can&garbage_can&wastebin&ash_bin&ashbin&dustbin&trash_barrel&trash_bin', 'bag&traveling_bag&travelling_bag&grip&suitcase', 'basket&handbasket', 'bathtub&bathing_tub&bath&tub', 'bed', 'bench', 'bicycle&bike&wheel&cycle', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus&autobus&coach&charabanc&double_decker&jitney&motorbus&motorcoach&omnibus&passenger_vehi', 'cabinet', 'camera&photograph_camera', 'can&tin&tin_can', 'cap', 'car&auto&automobile&machine&motorcar', 'chair', 'clock', 'computer_keyboard&keypad', 'dishwasher&dish_washer&dishwashing_machine', 'display&vedio_diaplay', 'earphone&earpiece&headphone&phone', 'faucet&spigot', 'file&file_cabinet&filing_cabinet', 'guitar', 'helmet', 'jar&vase', 'knife&dagger&bayonet', 'lamp&table lamp&floor lamp', 'laptop&laptop computer', 'loudspeaker&speaker&subwoofer&tweeter', 'mailbox&letter box', 'microphone&mike', 'microwave&oven', 'motorcycle&bike', 'mug', 'piano&pianoforte&forte_piano', 'pillow', 'pistol&handgun&side arm&shooting iron', 'pot&flowerpot', 'printer&printing_machine', 'remote_control&remote', 'rifle', 'rocket&projectile', 'skateboard', 'sofa&couch&lounge', 'stove', 'table', 'telephone&phone&telephone_set', 'tower', 'train&railroad_train', 'vessel&watercraft', 'washer&automatic_washer&washing_machine']
        
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                # self.net = models.resnet18(pretrained=self.pretraining)
                # self.net.fc = nn.Linear(512,nclasses)
                # if attention == 'TripletAttention':
                if attention != None:
                    self.net = ResidualNet("ImageNet", 18, nclasses, attention, pretrain=pretraining, triplet_pos = triplet_pos, pretrain_add = pretrain_add)#还需要记载预训练模型
                else:
                    self.net = ResidualNet("ImageNet", 18, nclasses, pretrain=pretraining)#还需要记载预训练模型
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet50':
                # self.net = models.resnet50(pretrained=self.pretraining)
                # self.net.fc = nn.Linear(2048,nclasses)
                # if attention == 'TripletAttention':
                if attention != None:
                    self.net = ResidualNet("ImageNet", 50, nclasses, attention, pretrain=pretraining, triplet_pos = triplet_pos, pretrain_add = pretrain_add)#还需要记载预训练模型
                else:
                    self.net = ResidualNet("ImageNet", 50, nclasses, pretrain=pretraining)#还需要记载预训练模型
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=55, cnn_name='vgg11', num_views=12, lossv = False):
        super(MVCNN, self).__init__(name)
        # 15class
        # self.classnames=['airplane','bench','bookshelf','bowl','chair','cone','door','guitar','keyboard','laptop','person','sofa','stairs','stool','xbox']
       
        # self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        # self.classnames = ['airplane&aeroplane&plane', 'ashcan&trash_can&garbage_can&wastebin&ash_bin&ashbin&dustbin&trash_barrel&trash_bin', 'bag&traveling_bag&travelling_bag&grip&suitcase', 'basket&handbasket', 'bathtub&bathing_tub&bath&tub', 'bed', 'bench', 'bicycle&bike&wheel&cycle', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus&autobus&coach&charabanc&double_decker&jitney&motorbus&motorcoach&omnibus&passenger_vehi', 'cabinet', 'camera&photograph_camera', 'can&tin&tin_can', 'cap', 'car&auto&automobile&machine&motorcar', 'chair', 'clock', 'computer_keyboard&keypad', 'dishwasher&dish_washer&dishwashing_machine', 'display&vedio_diaplay', 'earphone&earpiece&headphone&phone', 'faucet&spigot', 'file&file_cabinet&filing_cabinet', 'guitar', 'helmet', 'jar&vase', 'knife&dagger&bayonet', 'lamp&table lamp&floor lamp', 'laptop&laptop computer', 'loudspeaker&speaker&subwoofer&tweeter', 'mailbox&letter box', 'microphone&mike', 'microwave&oven', 'motorcycle&bike', 'mug', 'piano&pianoforte&forte_piano', 'pillow', 'pistol&handgun&side arm&shooting iron', 'pot&flowerpot', 'printer&printing_machine', 'remote_control&remote', 'rifle', 'rocket&projectile', 'skateboard', 'sofa&couch&lounge', 'stove', 'table', 'telephone&phone&telephone_set', 'tower', 'train&railroad_train', 'vessel&watercraft', 'washer&automatic_washer&washing_machine']
        self.nclasses = nclasses
        self.num_views = num_views
        self.lossv = lossv

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            # print("self.net_1")
            # print(self.net_1)
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(bs,12,512,7,7)
        
        if self.lossv is False:
            y = torch.max(y,1)[0].view(y.shape[0],-1)#(bs, 512*7*7)
            return y, self.net_2(y)#特征和分类结果
        else:
            y1 = torch.max(y,1)[0].view(y.shape[0],-1)#(bs, 512*7*7)
            return y1, self.net_2(y1), self.net_2(y.view(y.shape[0]*y.shape[1],-1))#(bs, 512*7*7),(bs, nclass),(bs*12, nclass)


        # return (self.net_2(torch.max(y,1)[0].view(y.shape[0],-1)), self.net_2(y.view(y.shape[0]*y.shape[1],-1)))

