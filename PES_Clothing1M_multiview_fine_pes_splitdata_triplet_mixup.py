import os
import os.path
import argparse
import random
import time
import numpy as np
import os
from PIL import Image
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torch.backends.cudnn as cudnn
import sys
import torch.nn.functional as F
from common.tools import AverageMeter, getTime,  evaluateWithBoth, ProgressMeter
from MVCNN.MVCNN import SVCNN, MVCNN

from FINE.svd_classifier import *
from FINE.gmm import *
from FINE.util import *

from gceloss import *
from loss_mixup import Mixup

from PES_Clothing1M_singleview_fine_pes_triplet import train_singleview

from loss_view import loss_view

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'\
print("12/12 modify")

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_percent', default=1, type=float, help='data number percent')
parser.add_argument('--batch_size', default=20, type=int, help='batchsize')
parser.add_argument('--lr', '--learning_rate', default=5e-4, type=float, help='initial learning rate')#T1
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=0.001)
parser.add_argument('--workers', type=int, help='number of data loading workers (default: 8)', default=8)
parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--num_views", type=int, help="number of views", default=12)
parser.add_argument('--pretrain', type=str, help='pretrain', default='Yes')
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/att/bam')
parser.add_argument('--num_iters_epoch', default=10, type=int)

parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--num_epochs', default=100, type=int) #150  80
parser.add_argument('--T1', default=5, type=int)# 50  20
parser.add_argument('--T2', default=40, type=int, help='default 5')# 30 10
parser.add_argument('--PES_lr', default=5e-5, type=float, help='initial learning rate')# T2 5e-6
parser.add_argument('--num_networks', default=1, type=int)

parser.add_argument('--num_classes', type=int, default=40)
parser.add_argument("-cleandata_path", type=str, default="/home/gq/data/3DWebNet40_dataset/train_clean") #干净数据
parser.add_argument("-bgndata_path", type=str, default="/home/gq/data/3DWebNet40_dataset/train_bg_noise") #背景噪声数据
parser.add_argument("-lndata_path", type=str, default="/home/gq/data/3DWebNet40_dataset/train_label_noise") #标签噪声数据
parser.add_argument("-checkdata", type=bool, default=True)
parser.add_argument('--train_path',  default='/home/gq/data/3DWebNet40_dataset/train', type=str, help='the train data path')
parser.add_argument('--test_path',  default='/home/gq/data/3DWebNet40_dataset/test', type=str, help='the test data path')

parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("-resume", "--resume", type=str, help="resume", default = None)


#label
parser.add_argument("-label_conf", "--label_conf", type=float, help="the label select p_threshold", default=0)

#fine
parser.add_argument("-distill_mode", "--distill_mode", type=str, help="distill_mode", default="fine-gmm")#clothing1M
parser.add_argument("-every", "--every", type=int, help="the update frequency of data", default=10)
parser.add_argument("-p_threshold", "--p_threshold", type=float, help="the select p_threshold", default=0.5)

# mixup
parser.add_argument("--alpha", type=float, help="the alpha os mixup", default=0.05)

#print
parser.add_argument("--print_frequency", type=int, help="the print frequency of data", default=300)

# split data train
parser.add_argument("--select_type", type=int, help="the type of selecting data", default=0)#挑选数据集的标准 0表示联合筛选且划分数据集，1表示仅标签筛选，2表示仅特征筛选， 3表示联合筛选不划分数据集
parser.add_argument("--lamda", type=float, help="the coefficient of edge loss of edge data", default=1)#edge数据集的loss参数
parser.add_argument("--theta", type=float, help="the coefficient of all loss of mixup data", default=1)#mixup数据集的loss参数
parser.add_argument("--mixtype", type=int, help="the type of mixup", default=0)#0表示所有的全部mixup 1表示mixup的是clean+edge 2表示只使用mixup不划分数据集  3表示只使用C和E不使用mixup
parser.add_argument("--lossv", type=bool, help="add view loss", default = False) #是否添加一致性loss
parser.add_argument("--beta", type=float, help="the coefficient of view loss", default = 1) #一致性loss的参数

# att type
parser.add_argument("--att_type", type=int, help="attention type", default=0)#0是triplet_attention 1是SeNet  2表示CBAM 3表示BAM

# triplet pos
parser.add_argument("--triplet_pos", type=int, help="where the triplet attention at", default=1)#1是在每个block的最后一个加上 0是所有的block全都加上 2只在最后一层加triplet  3表示不加注意力机制

# triplet single pretrain
parser.add_argument("--singleview_pretrain", type=bool, help="whether need single view pretraining", default=True)
parser.add_argument("--singleview_pretrain_epoch", type=int, help="single view pretraining epoch", default=5)
parser.add_argument("--pretrain_add", type=int, help="pretrain_add", default=1)# 0表示使用triplet官方预训练模型、1表示仅使用res50预训练模型


#test mode
parser.add_argument("--test_mode", type=bool, help="test_mode", default=False)


args = parser.parse_args()
print(args)
# print("triplet不用官方模型 再试一次")
# os.system('nvidia-smi')

if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

# define gpu id
args.n_gpu = torch.cuda.device_count()
print('args.n_gpu = ', args.n_gpu)
if(args.n_gpu > 0):
    device = torch.device('cuda', args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class Clothing1M_Dataset_Multiview(Dataset):
    def __init__(self, data, labels, root_dir, transform=None, target_transform=None):
        # self.length = (len(self.train_labels) / args.num_views)
        # self.train_data = []
        # self.train_labels = []
        # for i in range(self.length):
        #     grep = []
        #     for iview in range(self.num_views):
        #         grep.append(data[i * self.num_views + iview])
        #     self.train_labels.append(labels[i * self.num_views])
        #     self.train_data.append(grep)
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.root_dir = root_dir    
        self.length = len(self.train_labels)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.target_transform = target_transform
        print("NewDataset path:", self.root_dir)
        print("NewDataset length:", self.length)

    def __getitem__(self, index):
        img_paths, target = self.train_data[index], self.train_labels[index]
        imgs = []
        for path in img_paths:
            img_path = os.path.join(self.root_dir, path)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        if self.target_transform is not None:
            target = self.target_transform(target)         
        return torch.stack(imgs),target 

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


class Clothing1M_Unlabeled_Dataset_Multiview(Dataset):
    def __init__(self, data, root_dir, transform=None):
        # self.length = (len(self.train_labels) / args.num_views)
        # self.train_data = []
        # for i in range(self.length):
        #     grep = []
        #     for iview in range(self.num_views):
        #         grep.append(data[i * self.num_views + iview])
        #     self.train_data.append(grep)
        self.train_data = np.array(data)
        self.root_dir = root_dir  
        self.length = len(self.train_data)      

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.target_transform = target_transform
        print("NewDataset length:", self.length)

    def __getitem__(self, index):
        img_paths = self.train_data[index]
        imgs1 = []
        imgs2 = []
        for path in img_paths:
            img_path = os.path.join(self.root_dir, path)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img1 = self.transform(img)
                img2 = self.transform(img)
            imgs1.append(img1)
            imgs2.append(img2)        
        return torch.stack(imgs1),torch.stack(imgs2)

    def __len__(self):
        return self.length


def create_model(pretrained):
    if(pretrained == 'Yes'):
        pretrain = True
    else:
        pretrain = False

    # 0是triplet_attention 1是SeNet  2表示CBAM 3表示BAM
    att_list = {0 : "TripletAttention", 1: "SENet", 2:"CBAM", 3:"BAM"} 
    
    if args.triplet_pos == 3:        
        att_type = None
    else:
        # att_type = "TripletAttention"
        att_type = att_list[args.att_type]
    
    cnet = SVCNN("SVCNN", nclasses=args.num_classes, pretraining=pretrain, cnn_name=args.cnn_name, attention=att_type, triplet_pos = args.triplet_pos, pretrain_add = args.pretrain_add)
    if torch.cuda.is_available:
        cnet = cnet.to(device)
    print(cnet)
    if args.singleview_pretrain and args.resume == None:
        cnet = train_singleview(cnet, args.train_path, args.test_path, classes_to_idx, args.singleview_pretrain_epoch, device, os.path.join(args.model_dir, 'Singleview'))
        #加载路径：/root/data1/code/PES-main/save_models/webnet40/half_test/singlepre_add1_0.8_mixup11_4/Singleview/checkpoint.pth
    model = MVCNN("MVCNN", cnet, nclasses=args.num_classes, cnn_name=args.cnn_name, num_views=args.num_views, lossv = args.lossv)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    if torch.cuda.is_available:
        model = model.to(device)
    return model


def splite_confident(outs, noisy_targets, portion_index=None):
    probs, preds = torch.max(outs.data, 1)

    confident_indexs = []
    unconfident_indexs = []
    for i in range(0, len(noisy_targets)):
        if preds[i] == noisy_targets[i]:# and probs[i] > args.label_conf:
            if portion_index is None:
                confident_indexs.append(i)
            else:
                confident_indexs.append(portion_index[i])
        else:
            if portion_index is None:
                unconfident_indexs.append(i)
            else:
                unconfident_indexs.append(portion_index[i])

    print("\t",getTime(), "confident and unconfident num:", len(confident_indexs), len(unconfident_indexs))
    return confident_indexs, unconfident_indexs

def splite_confident_2(outs, noisy_targets, portion_index=None):
    probs, preds = torch.max(outs.data, 1)

    confident_indexs = []
    pseudo_indexs = []
    pseudo_labels = []
    unconfident_indexs = []
    for i in range(0, len(noisy_targets)):
        if preds[i] == noisy_targets[i]:
            if portion_index is None:
                confident_indexs.append(i)
            else:
                confident_indexs.append(portion_index[i])
        else:
            if probs[i] > args.label_conf:
                if portion_index is None:
                    pseudo_indexs.append(i)
                else:
                    pseudo_indexs.append(portion_index[i])
                pseudo_labels.append(probs[i])
            else:
                if portion_index is None:
                    unconfident_indexs.append(i)
                else:
                    unconfident_indexs.append(portion_index[i])

    print("\t",getTime(), "confident and unconfident num:", len(confident_indexs), len(unconfident_indexs))
    print("\t",getTime(), "pseudo num:", len(pseudo_indexs))
    print(len(pseudo_indexs))
    # print(pseudo_indexs)
    return confident_indexs, unconfident_indexs, pseudo_indexs, pseudo_labels

def update_trainloader(model, train_data, noisy_targets, fixed_confident_indexs=None):
    print("updating trainloader...")
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)

    confident_indexs, unconfident_indexs = splite_confident(soft_outs, noisy_targets)
    _, preds = torch.max(soft_outs.data, 1)

    train_data = np.array(train_data)
    train_dataset = Clothing1M_Dataset_Multiview(train_data[confident_indexs], preds[confident_indexs], args.train_path, transform, target_transform)
    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_classes, dtype=int)
    for item in preds[confident_indexs]:
        train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    print("train categroy mean:", np.mean(train_nums, dtype=int), ", category:", train_nums, ", precent:", np.mean(train_nums) / train_nums)
    # class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).to(device)
    
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0

    class_weights = torch.FloatTensor(cw).to(device)
    # class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums)
    ceriation = nn.CrossEntropyLoss(weight=class_weights).to(device)

    return trainloader, None, ceriation

def check_selectdata_pes(model, train_data, noisy_targets, cleandata_data, cleandata_labels):
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)

    confident_indexs, unconfident_indexs = splite_confident(soft_outs, noisy_targets)
    _, preds = torch.max(soft_outs.data, 1)
    # print("cleandata = ",cleandata_data[:10])
    right_num = 0
    num = 0
    for conidx in confident_indexs:
        try:
            # print("train_data[conidx] = ",train_data[conidx])
            # print("preds[conidx] = ", preds[conidx].item())
            idx = cleandata_data.index(train_data[conidx][0])
            num = num + 1
            # print("idx = ",idx)
            # print("cleandata_labels[idx] = ", cleandata_labels[idx])
            
            if cleandata_labels[idx] == preds[conidx]:
                right_num = right_num + 1
        except:  
            # print("not found")             
            continue
    # print("confident and unconfident num:", len(confident_indexs),"  ", len(unconfident_indexs))    
    # print("num = ", num)
    print("right num: ", right_num, ", rightnum / confident num:", round(right_num / len(confident_indexs), 3))     
    return 


def noisy_refine(model, train_loader, refine_ceriation, refine_times):
    if refine_times <= 0:
        return model
    # frezon all layers and add a new final layer
    print("Begin to refine network...")
    for param in model.parameters():
        param.requires_grad = False

    model.net_2 = nn.Linear(2048, args.num_classes)
    model = model.to(device)

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.PES_lr)    
    for iter_index in range(refine_times):
        train_iter = iter(train_loader)
        # train(model, train_iter, refine_ceriation, optimizer_adam, args.num_iters_epoch)
        print("[Pes train] ", iter_index)
        train(model, train_iter, refine_ceriation, optimizer_adam)
        _, train_acc = evaluate(model, train_loader, refine_ceriation, "PES Train Acc:")
        _, test_acc = evaluate(model, test_loader, refine_ceriation, "PES Test Acc:")
        # check_selectdata(model, whole_train_data, whole_train_labels, cleandata_data, cleandata_labels)
        # if best_test_acc < test_acc:
        #     torch.save(model1.state_dict(), filepath1_2)
        #     best_test_acc = test_acc

    for param in model.parameters():
        param.requires_grad = True

    return model


def update_trainloader_fine(model, origin_train_data, origin_train_labels):
    print("updating trainloader fine...")   

    # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle？ ！！！
    origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
    origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=4, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    print("\tgetting features...")
    current_features, current_labels = get_features(model, origin_data_loader, device)
    print("\tget features end")
    # datanum = len(current_labels)
    prev_features, prev_labels = current_features, current_labels 
    print("\tfining")
    teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=0.5, norm=True)
    print("\tfine end")
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)
    cur_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[teacher_idx], origin_train_labels[teacher_idx], args.train_path, transform, target_transform)
    cur_trainloader = DataLoader(dataset=cur_train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)    
    print("updating trainloader fine end, the select number is ", len(teacher_idx)) 
    return cur_trainloader, teacher_idx

def update_trainloader_fine_pes(model, origin_train_data, origin_train_labels):
    print("updating trainloader fine&PES...") 

    print("\tupdating trainloader fine...")
    #fine
    # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
    origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
    origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=4, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    print("\t\tgetting features...")
    current_features, current_labels = get_features(model, origin_data_loader, device)
    print("\t\tget features end")
    # datanum = len(current_labels)
    prev_features, prev_labels = current_features, current_labels 
    print("\t\tfining")
    teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=0.5, norm=True)
    print("\t\tfine end")
    
    # cur_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[teacher_idx], origin_train_labels[teacher_idx], args.train_path, transform, target_transform)
    # cur_trainloader = DataLoader(dataset=cur_train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)    
    print("\tupdating trainloader fine end, the select number is ", len(teacher_idx)) 

    print("\tupdating trainloader PES...")
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)
    confident_indexs, unconfident_indexs = splite_confident(soft_outs, origin_train_labels)
    _, preds = torch.max(soft_outs.data, 1)

    print("\tfusing data...")
    indexes_set = set(teacher_idx) | set(confident_indexs)
    indexes = list(indexes_set)

    labels = []
    for idx in indexes:
        # print(idx)
        if idx in list(teacher_idx):            
            # print(origin_train_labels[idx])
            labels.append(origin_train_labels[idx])
        else:
            labels.append(preds[idx])
            # print(preds[idx])
    origin_train_data = np.array(origin_train_data)
    labels = np.array(labels)
    train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[indexes], labels, args.train_path, transform, target_transform)
    cur_trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_classes, dtype=int)
    for item in labels:
        train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    print("train categroy mean:", np.mean(train_nums, dtype=int), ", category:", train_nums, ", precent:", np.mean(train_nums) / train_nums)
    # class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).to(device)
    
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0

    class_weights = torch.FloatTensor(cw).to(device)
    # class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums)
    ceriation = nn.CrossEntropyLoss(weight=class_weights).to(device)

    return cur_trainloader, ceriation

def update_trainloader_fine_pes_split(model, origin_train_data, origin_train_labels):
    print("updating trainloader fine&PES...") 

    print("\tupdating trainloader fine...")
    #fine
    # 因���这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
    origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
    origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=int(args.batch_size / 3), num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    print("\t\tgetting features...")
    current_features, current_labels = get_features(model, origin_data_loader, device)
    print("\t\tget features end")
    # datanum = len(current_labels)
    prev_features, prev_labels = current_features, current_labels 
    print("\t\tfining")
    teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=args.p_threshold, norm=True)
    print("\t\tfine end, the selected data number is ", len(teacher_idx))

    print("\tupdating trainloader PES...")
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)
    confident_indexs, unconfident_indexs = splite_confident(soft_outs, origin_train_labels)
    _, preds = torch.max(soft_outs.data, 1) 

    print("\tspliting data...")
    #干净数据取交集
    clean_indexes_set = set(teacher_idx) & set(confident_indexs)
    clean_indexes = list(clean_indexes_set)

    #边缘数据集
    edge_indexes_set = (set(teacher_idx) | set(confident_indexs)) - clean_indexes_set
    edge_indexes = list(edge_indexes_set)

    print("\tsplited")
    print("the num of clean data is ", len(clean_indexes))
    print("the num of edge data is ", len(edge_indexes))
    # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
    # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
    print("lamda is ", args.lamda)

    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    clean_labels = np.array(origin_train_labels[clean_indexes])
    clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
    cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    
    edge_labels = np.array(origin_train_labels[edge_indexes])
    edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
    cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # Loss function
    clean_train_nums = np.zeros(args.num_classes, dtype=int)
    edge_train_nums = np.zeros(args.num_classes, dtype=int)
    for item in clean_labels:
        clean_train_nums[item] += 1
    for item in edge_labels:
        edge_train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
    # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
    
    with np.errstate(divide='ignore'):
        cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
        cw_clean[cw_clean == np.inf] = 0
        cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
        cw_edge[cw_edge == np.inf] = 0
    print("clean train weight:", cw_clean)
    print("edge train weight:", cw_clean)
    clean_class_weights = torch.FloatTensor(cw_clean).to(device)
    clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
    edge_class_weights = torch.FloatTensor(cw_edge).to(device)
    edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
    # edge_ceriation(loigts, target)

    return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation

def check_selectdata_fine_pes_split(model, origin_train_data, origin_train_labels, cleandata_data, cleandata_labels):

    print("\tupdating trainloader fine...")
    #fine
    # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
    origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
    origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=1, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    # print("\t\tgetting features...")
    current_features, current_labels = get_features(model, origin_data_loader, device)
    # print("\t\tget features end")
    # datanum = len(current_labels)
    prev_features, prev_labels = current_features, current_labels 
    # print("\t\tfining")
    teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=args.p_threshold, norm=True)
    print("\t\tfine end, the selected data number is ", len(teacher_idx))

    print("\tupdating trainloader PES...")
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(origin_train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)
    confident_indexs, unconfident_indexs = splite_confident(soft_outs, origin_train_labels)
    _, preds = torch.max(soft_outs.data, 1)

    print("\tspliting data...")
    #干净数据取交集
    clean_indexes_set = set(teacher_idx) & set(confident_indexs)
    clean_indexes = list(clean_indexes_set)

    #边缘数据集
    edge_indexes_set = (set(teacher_idx) | set(confident_indexs)) - clean_indexes_set
    edge_indexes = list(edge_indexes_set)
    
    clean_right_num = 0
    num = 0
    for conidx in clean_indexes:
        try:
            idx = cleandata_data.index(train_data[conidx][0])
            num = num + 1            
            if cleandata_labels[idx] == preds[conidx]:
                clean_right_num = clean_right_num + 1
        except:             
            continue    
    print("the num of clean data is ", len(clean_indexes))
    print("the num in cleandata ", num)
    print("clean_right num: ", clean_right_num, ", rightnum / clean_indexes num:", round(clean_right_num / len(clean_indexes), 3)) 

    edge_right_num = 0
    num = 0
    for conidx in edge_indexes:
        try:
            idx = cleandata_data.index(train_data[conidx][0])
            num = num + 1
            
            if cleandata_labels[idx] == preds[conidx]:
                edge_right_num = edge_right_num + 1
        except:             
            continue    
    print("the num of edge data is ", len(edge_indexes))
    print("the num in edgedata ", num)
    print("edge_right num: ", edge_right_num, ", rightnum / edge_indexes num:", round(edge_right_num / len(edge_indexes), 3)) 

    # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
    # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
    print("lamda is ", args.lamda)

    print("\t fusing data...")
    indexes_set = set(teacher_idx) | set(confident_indexs)
    indexes = list(indexes_set)
    right_num = 0
    num = 0
    for conidx in indexes:
        try:
            idx = cleandata_data.index(train_data[conidx][0])
            num = num + 1
            
            if cleandata_labels[idx] == preds[conidx]:
                right_num = right_num + 1
        except:             
            continue    
    print("the num of fusing data is ", len(indexes))
    print("the num in fusing data ", num)
    print("right num: ", right_num, ", rightnum / indexes num:", round(right_num / len(indexes), 3)) 

    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    clean_labels = np.array(origin_train_labels[clean_indexes])
    clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
    cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    
    edge_labels = np.array(origin_train_labels[edge_indexes])
    edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
    cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # Loss function
    clean_train_nums = np.zeros(args.num_classes, dtype=int)
    edge_train_nums = np.zeros(args.num_classes, dtype=int)
    for item in clean_labels:
        clean_train_nums[item] += 1
    for item in edge_labels:
        edge_train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
    # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
    
    with np.errstate(divide='ignore'):
        cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
        cw_clean[cw_clean == np.inf] = 0
        cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
        cw_edge[cw_edge == np.inf] = 0
    print("clean train weight:", cw_clean)
    print("edge train weight:", cw_clean)
    clean_class_weights = torch.FloatTensor(cw_clean).to(device)
    clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
    edge_class_weights = torch.FloatTensor(cw_edge).to(device)
    edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
    # edge_ceriation(loigts, target)

    return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation

def update_trainloader_fine_pes_split_half(model, origin_train_data, origin_train_labels):
    if args.select_type == 0:
        print("updating trainloader fine&PES and split dataloader...")
    elif args.select_type == 1:
        print("updating trainloader PES...")
    elif args.select_type == 2:
        print("updating trainloader fine...")
    elif args.select_type == 3:
        print("updating trainloader fine&PES...")

    if args.select_type != 1:
        print("\tupdating trainloader fine...")
        #fine
        # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
        origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
        origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=int(args.batch_size / 3), num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
        print("\t\tgetting features...")
        print(len(origin_data_loader))
        current_features, current_labels = get_features(model, origin_data_loader, device, lossv = args.lossv)
        print(len(current_features))
        print(len(current_labels))
        print("\t\tget features end")
        # datanum = len(current_labels)
        prev_features, prev_labels = current_features, current_labels 
        print("\t\tfining")
        teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=args.p_threshold, norm=True)
        print("\t\tfine end, the selected data number is ", len(teacher_idx))
    if args.select_type != 2:     
        print("\tupdating trainloader PES...")
        predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        soft_outs = predict_softmax(predict_loader, model, device)
        confident_indexs, unconfident_indexs, pseudo_indexs, pseudo_labels = splite_confident_2(soft_outs, origin_train_labels)
        _, preds = torch.max(soft_outs.data, 1) 

    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    if args.select_type == 0:
        # print("updating trainloader fine&PES and split dataloader...")

        print("\tspliting data...")
        #干净数据取交集
        clean_indexes_set = set(teacher_idx) & set(confident_indexs)
        clean_indexes = list(clean_indexes_set)

        #边缘数据集
        edge_indexes_set = (set(teacher_idx) | set(confident_indexs)) - clean_indexes_set
        edge_indexes = list(edge_indexes_set)

        print("\tsplited")
        print("the num of clean data is ", len(clean_indexes))
        print("the num of edge data is ", len(edge_indexes))
        # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
        # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
        print("lamda is ", args.lamda)

        #create dataloader

        clean_labels = np.array(origin_train_labels[clean_indexes])
        clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
        cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
        
        edge_labels = np.array(origin_train_labels[edge_indexes])
        edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
        cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

        # Loss function
        clean_train_nums = np.zeros(args.num_classes, dtype=int)
        edge_train_nums = np.zeros(args.num_classes, dtype=int)
        for item in clean_labels:
            clean_train_nums[item] += 1
        for item in edge_labels:
            edge_train_nums[item] += 1
        # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
        # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
        # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
        
        with np.errstate(divide='ignore'):
            cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
            cw_clean[cw_clean == np.inf] = 0
            cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
            cw_edge[cw_edge == np.inf] = 0
        print("clean train weight:", cw_clean)
        print("edge train weight:", cw_clean)
        clean_class_weights = torch.FloatTensor(cw_clean).to(device)
        clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
        edge_class_weights = torch.FloatTensor(cw_edge).to(device)
        edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
        # edge_ceriation(loigts, target)

        if args.mixtype != 1:
            return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation
        else:
            tmp_indexes = clean_indexes + edge_indexes
            tmp_labels = np.array(origin_train_labels[tmp_indexes])
            tmp_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[tmp_indexes], tmp_labels, args.train_path, transform, target_transform)
            cur_tmp_trainloader = DataLoader(dataset=tmp_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
        
            return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, cur_tmp_trainloader

    elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
        # print("updating trainloader PES...")
        
        #干净数据
        if args.select_type == 1:
            clean_indexes_set = set(confident_indexs)
        elif args.select_type == 2:
            clean_indexes_set = set(teacher_idx)
        elif args.select_type == 3:
            clean_indexes_set = (set(teacher_idx) | set(confident_indexs))

        clean_indexes = list(clean_indexes_set)

        print("the num of clean data is ", len(clean_indexes))

        #create dataloader
        clean_labels = np.array(origin_train_labels[clean_indexes])
        clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
        cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
        
        # Loss function
        clean_train_nums = np.zeros(args.num_classes, dtype=int)
        for item in clean_labels:
            clean_train_nums[item] += 1
        # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
        # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
        # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
        
        with np.errstate(divide='ignore'):
            cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
            cw_clean[cw_clean == np.inf] = 0
        print("clean train weight:", cw_clean)
        clean_class_weights = torch.FloatTensor(cw_clean).to(device)
        clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)

        return cur_clean_trainloader, clean_ceriation

def check_selectdata_fine_pes_split_half(model, origin_train_data, origin_train_labels, cleandata_data,  bgndata_data, clean_labels, bgndata_labels):
    if args.select_type == 0:
        print("updating trainloader fine&PES and split dataloader...")
    elif args.select_type == 1:
        print("updating trainloader PES...")
    elif args.select_type == 2:
        print("updating trainloader fine...")
    elif args.select_type == 3:
        print("updating trainloader fine&PES...")

    if args.select_type != 1:
        print("\tupdating trainloader fine...")
        #fine
        # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
        origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
        origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=int(args.batch_size / 3), num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
        print("\t\tgetting features...")
        print(len(origin_data_loader))
        current_features, current_labels = get_features(model, origin_data_loader, device, lossv = args.lossv)
        print(len(current_features))
        print(len(current_labels))
        print("\t\tget features end")
        # datanum = len(current_labels)
        prev_features, prev_labels = current_features, current_labels 
        print("\t\tfining")
        teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=args.p_threshold, norm=True)
        print("\t\tfine end, the selected data number is ", len(teacher_idx))
    if True:     
        print("\tupdating trainloader PES...")
        predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        soft_outs = predict_softmax(predict_loader, model, device)
        confident_indexs, unconfident_indexs, pseudo_indexs, pseudo_labels = splite_confident_2(soft_outs, origin_train_labels)
        _, preds = torch.max(soft_outs.data, 1) 
    
    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    #check
    #干净数据有哪些   
    # classes = list(classes_to_idx.keys())
    mix_data = cleandata_data + bgndata_data
    mix_labels = clean_labels + bgndata_labels

    if args.select_type == 0:
        # print("updating trainloader fine&PES and split dataloader...")

        print("\tspliting data...")
        #干净数据取交集
        clean_indexes_set = set(teacher_idx) & set(confident_indexs)
        clean_indexes = list(clean_indexes_set)

        #边缘数据集
        edge_indexes_set = (set(teacher_idx) | set(confident_indexs)) - clean_indexes_set
        edge_indexes = list(edge_indexes_set)

        #check
        num = 0
        clean_right_num = 0
        for conidx in clean_indexes:
            try:
                # print(conidx)
                # print(train_data[conidx][0])
                idx = mix_data.index(train_data[conidx][0])
                num = num + 1            
                if mix_labels[idx] == preds[conidx]:
                    clean_right_num += 1
            except:             
                continue 
        print("num = ", num)
        edge_right_num = 0
        for conidx in edge_indexes:
            try:
                idx = mix_data.index(train_data[conidx][0])
                num = num + 1            
                if mix_labels[idx] == preds[conidx]:
                    edge_right_num += 1
            except:             
                continue 

        print("\tsplited")
        print("the num of clean data is ", len(clean_indexes))
        print("the ratio is ", clean_right_num / len(clean_indexes))
        print("the num of edge data is ", len(edge_indexes))
        print("the ratio is ", edge_right_num / len(edge_indexes))
        # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
        # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
        print("lamda is ", args.lamda)

        #create dataloader

        clean_labels = np.array(origin_train_labels[clean_indexes])
        clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
        cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
        
        edge_labels = np.array(origin_train_labels[edge_indexes])
        edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
        cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

        # Loss function
        clean_train_nums = np.zeros(args.num_classes, dtype=int)
        edge_train_nums = np.zeros(args.num_classes, dtype=int)
        for item in clean_labels:
            clean_train_nums[item] += 1
        for item in edge_labels:
            edge_train_nums[item] += 1
        # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
        # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
        # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
        
        with np.errstate(divide='ignore'):
            cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
            cw_clean[cw_clean == np.inf] = 0
            cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
            cw_edge[cw_edge == np.inf] = 0
        print("clean train weight:", cw_clean)
        print("edge train weight:", cw_clean)
        clean_class_weights = torch.FloatTensor(cw_clean).to(device)
        clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
        edge_class_weights = torch.FloatTensor(cw_edge).to(device)
        edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
        # edge_ceriation(loigts, target)

        if args.mixtype != 1:
            return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation
        else:
            tmp_indexes = clean_indexes + edge_indexes
            tmp_labels = np.array(origin_train_labels[tmp_indexes])
            tmp_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[tmp_indexes], tmp_labels, args.train_path, transform, target_transform)
            cur_tmp_trainloader = DataLoader(dataset=tmp_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
        
            return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, cur_tmp_trainloader
    
    elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
        # print("updating trainloader PES...")
        
        #干净数据
        if args.select_type == 1:
            clean_indexes_set = set(confident_indexs)
        elif args.select_type == 2:
            clean_indexes_set = set(teacher_idx)
        elif args.select_type == 3:
            clean_indexes_set = (set(teacher_idx) | set(confident_indexs))

        clean_indexes = list(clean_indexes_set)

        #check
        num = 0
        clean_right_num = 0
        for conidx in clean_indexes:
            try:
                idx = mix_data.index(train_data[conidx][0])
                num = num + 1            
                if mix_labels[idx] == preds[conidx]:
                    clean_right_num += 1
            except:             
                continue 
        print("num = ", num)

        print("the num of clean data is ", len(clean_indexes))
        print("the ratio is ", clean_right_num / len(clean_indexes))

        #create dataloader
        clean_labels = np.array(origin_train_labels[clean_indexes])
        clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
        cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
        
        # Loss function
        clean_train_nums = np.zeros(args.num_classes, dtype=int)
        for item in clean_labels:
            clean_train_nums[item] += 1
        # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
        # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
        # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
        
        with np.errstate(divide='ignore'):
            cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
            cw_clean[cw_clean == np.inf] = 0
        print("clean train weight:", cw_clean)
        clean_class_weights = torch.FloatTensor(cw_clean).to(device)
        clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)

        return cur_clean_trainloader, clean_ceriation
    
    '''
    #check
    #干净数据有哪些   
    # classes = list(classes_to_idx.keys())
    mix_data = cleandata_data + bgndata_data
    mix_labels = clean_labels + bgndata_labels

    num = 0
    clean_right_num = 0
    for conidx in clean_indexes:
        try:
            # print(conidx)
            # print(train_data[conidx][0])
            idx = mix_data.index(train_data[conidx][0])
            num = num + 1            
            if mix_labels[idx] == preds[conidx]:
                clean_right_num += 1
        except:             
            continue 
    print("num = ", num)
    edge_right_num = 0
    for conidx in edge_indexes:
        try:
            idx = mix_data.index(train_data[conidx][0])
            num = num + 1            
            if mix_labels[idx] == preds[conidx]:
                edge_right_num += 1
        except:             
            continue 

    print("\tsplited")
    print("the num of clean data is ", len(clean_indexes))
    print("the ratio is ", clean_right_num / len(clean_indexes))
    print("the num of edge data is ", len(edge_indexes))
    print("the ratio is ", edge_right_num / len(edge_indexes))


    # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
    # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
    print("lamda is ", args.lamda)

    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    clean_labels = np.array(origin_train_labels[clean_indexes])
    clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
    cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    
    edge_labels = np.array(origin_train_labels[edge_indexes])
    edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
    cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)


    # Loss function
    clean_train_nums = np.zeros(args.num_classes, dtype=int)
    edge_train_nums = np.zeros(args.num_classes, dtype=int)
    for item in clean_labels:
        clean_train_nums[item] += 1
    for item in edge_labels:
        edge_train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
    # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
    
    with np.errstate(divide='ignore'):
        cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
        cw_clean[cw_clean == np.inf] = 0
        cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
        cw_edge[cw_edge == np.inf] = 0
    print("clean train weight:", cw_clean)
    print("edge train weight:", cw_clean)
    clean_class_weights = torch.FloatTensor(cw_clean).to(device)
    clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
    edge_class_weights = torch.FloatTensor(cw_edge).to(device)
    edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
    # edge_ceriation(loigts, target)

    if args.mixtype != 1:
        return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation
    else:
        tmp_indexes = clean_indexes + edge_indexes
        tmp_labels = np.array(origin_train_labels[tmp_indexes])
        tmp_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[tmp_indexes], tmp_labels, args.train_path, transform, target_transform)
        cur_tmp_trainloader = DataLoader(dataset=tmp_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    
        return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, cur_tmp_trainloader

    # return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation
    '''

def update_trainloader_fine_pes_split_2(model, origin_train_data, origin_train_labels):
    print("updating trainloader fine&PES...") 

    print("\tupdating trainloader fine...")
    #fine
    # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
    origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
    origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=int(args.batch_size / 3), num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    print("\t\tgetting features...")
    print(len(origin_data_loader))
    current_features, current_labels = get_features(model, origin_data_loader, device)
    print(len(current_features))
    print(len(current_labels))
    print("\t\tget features end")
    # datanum = len(current_labels)
    prev_features, prev_labels = current_features, current_labels 
    print("\t\tfining")
    teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=args.p_threshold, norm=True)
    print("\t\tfine end, the selected data number is ", len(teacher_idx))

    print("\tupdating trainloader PES...")
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)
    confident_indexs, unconfident_indexs, pseudo_indexs, pseudo_labels = splite_confident_2(soft_outs, origin_train_labels)
    _, preds = torch.max(soft_outs.data, 1) 

    print("\tspliting data...")
    #干净数据取交集
    clean_indexes_set = set(teacher_idx) & set(confident_indexs)
    clean_indexes = list(clean_indexes_set)

    #边缘数据集
    edge_indexes_set = (set(teacher_idx) | set(confident_indexs)) - clean_indexes_set
    edge_indexes = list(edge_indexes_set)

    #找到伪标签数据
    print(len(pseudo_indexs))
    # print(pseudo_indexs)
    print(len(current_features))
    print(len(current_features[0]))
    pseudo_current_features = [current_features[pseudo_indexs[i]] for i in range(0, len(pseudo_indexs)) if pseudo_indexs[i] < len(current_features)]
    pseudo_current_labels = pseudo_labels
    pseudo_prev_features, pseudo_prev_labels = pseudo_current_features, pseudo_current_labels 
    pseudo_indexes = fine(pseudo_current_features, pseudo_current_labels, fit=args.distill_mode, prev_features=pseudo_prev_features, prev_labels=pseudo_prev_labels, p_threshold=args.p_threshold, norm=True)
    

    print("\tsplited")
    print("the num of clean data is ", len(clean_indexes))
    print("the num of edge data is ", len(edge_indexes))
    print("the num of pseudo data is ", len(pseudo_indexes))
    # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
    # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
    print("lamda is ", args.lamda)

    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    clean_labels = np.array(origin_train_labels[clean_indexes])
    clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
    cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    
    edge_labels = np.array(origin_train_labels[edge_indexes])
    edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
    cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    pseudo_labels = np.array(pseudo_labels)
    pseudo_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[pseudo_indexes], pseudo_labels, args.train_path, transform, target_transform)
    cur_pseudo_trainloader = DataLoader(dataset=pseudo_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # Loss function
    clean_train_nums = np.zeros(args.num_classes, dtype=int)
    edge_train_nums = np.zeros(args.num_classes, dtype=int)
    for item in clean_labels:
        clean_train_nums[item] += 1
    for item in edge_labels:
        edge_train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
    # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
    
    with np.errstate(divide='ignore'):
        cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
        cw_clean[cw_clean == np.inf] = 0
        cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
        cw_edge[cw_edge == np.inf] = 0
    print("clean train weight:", cw_clean)
    print("edge train weight:", cw_clean)
    clean_class_weights = torch.FloatTensor(cw_clean).to(device)
    clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
    edge_class_weights = torch.FloatTensor(cw_edge).to(device)
    edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
    # edge_ceriation(loigts, target)

    return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, cur_pseudo_trainloader

def check_selectdata_fine_pes_split_2(model, origin_train_data, origin_train_labels):
    print("updating trainloader fine&PES...") 

    print("\tupdating trainloader fine...")
    #fine
    # 因为这里样本筛选是按照dataloader的顺序来挑的 所以dataloader不可以shuffle ！！！
    origin_data_dataset = Clothing1M_Dataset_Multiview(origin_train_data, origin_train_labels, args.train_path, transform, target_transform)
    origin_data_loader = DataLoader(dataset=origin_data_dataset, batch_size=int(args.batch_size / 3), num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)
    print("\t\tgetting features...")
    current_features, current_labels = get_features(model, origin_data_loader, device)
    print("\t\tget features end")
    # datanum = len(current_labels)
    prev_features, prev_labels = current_features, current_labels 
    print("\t\tfining")
    teacher_idx = fine(current_features, current_labels, fit=args.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=args.p_threshold, norm=True)
    print("\t\tfine end, the selected data number is ", len(teacher_idx))

    print("\tupdating trainloader PES...")
    predict_dataset = Clothing1M_Unlabeled_Dataset_Multiview(train_data, args.train_path, transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device)
    confident_indexs, unconfident_indexs, pseudo_indexs, pseudo_labels = splite_confident_2(soft_outs, origin_train_labels)
    _, preds = torch.max(soft_outs.data, 1) 

    print("\tspliting data...")
    #干净数据取交集
    clean_indexes_set = set(teacher_idx) & set(confident_indexs)
    clean_indexes = list(clean_indexes_set)

    #边缘数据集
    edge_indexes_set = (set(teacher_idx) | set(confident_indexs)) - clean_indexes_set
    edge_indexes = list(edge_indexes_set)

    #找到伪标签数据
    pseudo_current_features = current_features[pseudo_indexs]
    pseudo_current_labels = pseudo_labels
    pseudo_prev_features, pseudo_prev_labels = pseudo_current_features, pseudo_current_labels 
    pseudo_indexes = fine(pseudo_current_features, pseudo_current_labels, fit=args.distill_mode, prev_features=pseudo_prev_features, prev_labels=pseudo_prev_labels, p_threshold=args.p_threshold, norm=True)
    

    #check
    #干净数据有哪些   
    # classes = list(classes_to_idx.keys())
    mix_data = cleandata_data + bgndata_data
    mix_labels = clean_labels + bgndata_labels
    print(len(mix_data))
    print(mix_data)
    print(mix_labels)

    clean_right_num = 0
    for conidx in clean_indexes:
        try:
            idx = mix_data.index(train_data[conidx][0])
            num = num + 1            
            if mix_labels[idx] == preds[conidx]:
                clean_right_num += 1
        except:             
            continue 
    
    edge_right_num = 0
    for conidx in edge_indexes:
        try:
            idx = mix_data.index(train_data[conidx][0])
            num = num + 1            
            if mix_labels[idx] == preds[conidx]:
                edge_right_num += 1
        except:             
            continue 

    print("\tsplited")
    print("the num of clean data is ", len(clean_indexes))
    print("the ratio is ", clean_right_num / len(clean_indexes))
    print("the num of edge data is ", len(edge_indexes))
    print("the ratio is ", edge_right_num / len(edge_indexes))
    print("the num of pseudo data is ", len(pseudo_indexes))


    # args.lamda = len(clean_indexes) / (len(clean_indexes) + len(edge_indexes))
    # args.lamda = min(1, 1 - len(edge_indexes) / len(clean_indexes))
    print("lamda is ", args.lamda)

    #create dataloader
    origin_train_data = np.array(origin_train_data)
    origin_train_labels = np.array(origin_train_labels)

    clean_labels = np.array(origin_train_labels[clean_indexes])
    clean_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[clean_indexes], clean_labels, args.train_path, transform, target_transform)
    cur_clean_trainloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    
    edge_labels = np.array(origin_train_labels[edge_indexes])
    edge_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[edge_indexes], edge_labels, args.train_path, transform, target_transform)
    cur_edge_trainloader = DataLoader(dataset=edge_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    pseudo_labels = np.array(pseudo_labels)
    pseudo_train_dataset = Clothing1M_Dataset_Multiview(origin_train_data[pseudo_indexes], pseudo_labels, args.train_path, transform, target_transform)
    cur_pseudo_trainloader = DataLoader(dataset=pseudo_train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # Loss function
    clean_train_nums = np.zeros(args.num_classes, dtype=int)
    edge_train_nums = np.zeros(args.num_classes, dtype=int)
    for item in clean_labels:
        clean_train_nums[item] += 1
    for item in edge_labels:
        edge_train_nums[item] += 1
    # print("train categroy mean", np.mean(train_nums, dtype=int), "category", train_nums, "precent", np.mean(train_nums) / train_nums)
    # print("clean train categroy mean:", np.mean(clean_train_nums, dtype=int), ", category:", clean_train_nums, ", precent:", np.mean(clean_train_nums) / clean_train_nums)
    # print("edge train categroy mean:", np.mean(edge_train_nums, dtype=int), ", category:", edge_train_nums, ", precent:", np.mean(edge_train_nums) / clean_train_nums)
    
    with np.errstate(divide='ignore'):
        cw_clean = np.mean(clean_train_nums[clean_train_nums != 0]) / clean_train_nums
        cw_clean[cw_clean == np.inf] = 0
        cw_edge = np.mean(edge_train_nums[edge_train_nums != 0]) / edge_train_nums
        cw_edge[cw_edge == np.inf] = 0
    print("clean train weight:", cw_clean)
    print("edge train weight:", cw_clean)
    clean_class_weights = torch.FloatTensor(cw_clean).to(device)
    clean_ceriation = nn.CrossEntropyLoss(weight=clean_class_weights).to(device)
    edge_class_weights = torch.FloatTensor(cw_edge).to(device)
    edge_ceriation = GCELoss(edge_class_weights, num_classes = args.num_classes).to(device)
    # edge_ceriation(loigts, target)

    return cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, cur_pseudo_trainloader

def check_selectdata_fine(origin_train_data, origin_train_labels, cleandata_data, cleandata_labels, select_idx):
    right_num = 0
    for conidx in select_idx:
        try:
            # print("train_data[conidx] = ",train_data[conidx])
            # print("preds[conidx] = ", preds[conidx].item())
            idx = cleandata_data.index(origin_train_data[conidx][0])
            right_num = right_num + 1
            # print("idx = ",idx)
            # print("cleandata_labels[idx] = ", cleandata_labels[idx])
        except:  
            # print("not found")             
            continue
    # print("confident and unconfident num:", len(confident_indexs),"  ", len(unconfident_indexs))    
    # print("num = ", num)
    print("right num: ", right_num, ", rightnum / confident num:", round(right_num / len(select_idx), 3))     
    return 

def train(model, train_loader, ceriation, train_optimizer, isMixup = False, loss_view_cer = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(iter(train_loader)),
        [batch_time, data_time, losses, top1], prefix="Train ")

    end = time.time()
     # train one epoch
    logits = None
    in_data = None
    for i, (input, target) in enumerate(train_loader):
        target = Variable(target).to(device).long()
        N,V,C,H,W = input.size()
        in_data = Variable(input).view(-1,C,H,W).to(device)
        # in_data = Variable(input.to(device))

        model.train()
        
        data_time.update(time.time() - end)
        train_optimizer.zero_grad()

        if args.mixtype == 2 and isMixup:
            input = input.to(device)
            logits, loss = ceriation(input, target, model)#BCEloss
        # else:
        #     _, logits = model(in_data)
        #     loss = ceriation(logits, target)
        else:
            if args.lossv:
                _, logits, logits_view = model(in_data)
                # loss = ceriation(logits, target) + args.beta * loss_view_cer(logits_view)
                loss = ceriation(logits, target)
            else:
                _, logits = model(in_data) 
                loss = ceriation(logits, target)

        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0) * input.size(1))
        top1.update(acc1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # print("i = ",i)
        if (i+1) % args.print_frequency == 0:
            progress.display(i)
    return top1.avg, losses.avg

# 用于训练筛选后的数据,只使用c和e
def train_second(model, clean_trainloader, clean_ceriation, edge_trainloader, edge_ceriation, loss_view_cer = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(iter(clean_trainloader)),
        [batch_time, data_time, losses, top1], prefix="Train clean ")
    end = time.time()

    # train clean data for one epoch
    logits = None
    in_data = None
    for i, (input, target) in enumerate(clean_trainloader):
        target = Variable(target).to(device).long()
        N,V,C,H,W = input.size()
        in_data = Variable(input).view(-1,C,H,W).to(device)

        model.train()
        data_time.update(time.time() - end)
        train_optimizer.zero_grad()

        # _, logits = model(in_data)
        # loss = clean_ceriation(logits, target)
        if args.lossv:
            _, logits, logits_view = model(in_data)
            loss = clean_ceriation(logits, target) + args.beta * loss_view_cer(logits_view)
        else:
            _, logits = model(in_data) 
            loss = clean_ceriation(logits, target)

        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0) * input.size(1))
        top1.update(acc1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_frequency == 0:
            progress.display(i)
    
    # train edge data for one epoch
    args.lamda = 1
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')    
    progress= ProgressMeter(
        len(iter(edge_trainloader)),
        [batch_time, data_time, losses, top1], prefix="Train edge")
    
    logits = None
    in_data = None
    for i, (input, target) in enumerate(edge_trainloader):
        target = Variable(target).to(device).long()
        N,V,C,H,W = input.size()
        in_data = Variable(input).view(-1,C,H,W).to(device)

        model.train()
        data_time.update(time.time() - end)
        train_optimizer.zero_grad()

        # _, logits = model(in_data)
        # loss = edge_ceriation(logits, target) * args.lamda
        if args.lossv:
            _, logits, logits_view = model(in_data)
            # loss = loss = (edge_ceriation(logits, target) + args.beta * loss_view_cer(logits_view)) * args.lamda
            loss = edge_ceriation(logits, target) * args.lamda
        else:
            _, logits = model(in_data)
            loss = edge_ceriation(logits, target) * args.lamda

        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0) * input.size(1))
        top1.update(acc1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_frequency == 0:
            progress.display(i)

    return top1.avg, losses.avg

#c e 和 h都用
def train_third(model, trainloader, ceriation, clean_trainloader, clean_ceriation, edge_trainloader, edge_ceriation, train_optimizer, loss_view_cer = None):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(iter(clean_trainloader)),
        [batch_time, data_time, losses, top1], prefix="Train clean ")
    end = time.time()

    # train clean data for one epoch
    logits = None
    in_data = None
    for i, (input, target) in enumerate(clean_trainloader):
        target = Variable(target).to(device).long()
        N,V,C,H,W = input.size()
        in_data = Variable(input).view(-1,C,H,W).to(device)

        model.train()
        data_time.update(time.time() - end)
        train_optimizer.zero_grad()

        # _, logits = model(in_data)
        # loss = clean_ceriation(logits, target)
        if args.lossv:
            _, logits, logits_view = model(in_data)
            loss = clean_ceriation(logits, target) + args.beta * loss_view_cer(logits_view)
        else:
            _, logits = model(in_data) 
            loss = clean_ceriation(logits, target)

        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0) * input.size(1))
        top1.update(acc1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_frequency == 0:
            progress.display(i)
    
    # train edge data for one epoch
    # args.lamda = 0.5
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')    
    progress= ProgressMeter(
        len(iter(edge_trainloader)),
        [batch_time, data_time, losses, top1], prefix="Train edge")
    end = time.time()

    logits = None
    in_data = None
    for i, (input, target) in enumerate(edge_trainloader):
        target = Variable(target).to(device).long()
        N,V,C,H,W = input.size()
        in_data = Variable(input).view(-1,C,H,W).to(device)

        model.train()
        data_time.update(time.time() - end)
        train_optimizer.zero_grad()

        # _, logits = model(in_data)
        # loss = edge_ceriation(logits, target) * args.lamda
        if args.lossv:
            _, logits, logits_view = model(in_data)
            # loss = loss = (edge_ceriation(logits, target) + args.beta * loss_view_cer(logits_view)) * args.lamda
            loss = edge_ceriation(logits, target) * args.lamda
        else:
            _, logits = model(in_data)
            loss = edge_ceriation(logits, target) * args.lamda

        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0) * input.size(1))
        top1.update(acc1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_frequency == 0:
            progress.display(i)
    
    # train all data for one epoch
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')    
    progress= ProgressMeter(
        len(iter(trainloader)),
        [batch_time, data_time, losses, top1], prefix="Train all")
    end = time.time()

    logits = None
    in_data = None
    for i, (input, target) in enumerate(trainloader):
        target = Variable(target).to(device).long()
        input = input.to(device)
        # N,V,C,H,W = input.size()
        # in_data = Variable(input).view(-1,C,H,W).to(device)

        model.train()
        # _, logits = model(in_data)
        logits, loss = ceriation(input, target, model)#BCEloss

        loss =  loss* args.theta
        data_time.update(time.time() - end)
        train_optimizer.zero_grad()


        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0) * input.size(1))
        top1.update(acc1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_frequency == 0:
            progress.display(i)

    return top1.avg, losses.avg

def evaluate(model, eva_loader, ceriation, prefix, ignore=-1):
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            N,V,C,H,W = images.size()
            images = Variable(images).view(-1,C,H,W).to(device)
        # in_data = Variable(input.to(device))
            labels = Variable(labels).to(device).long()
            data_time.update(time.time() - end)

            # _, logist = model(images)
            if args.lossv:
                _, logist, _ = model(images)
            else:
                _, logist = model(images)

            loss = ceriation(logist, labels)
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0) / args.num_views)
            batch_time.update(time.time() - end)
            end = time.time()

    if prefix != "":
        # print("[Test] ",getTime(), prefix, round(top1.avg.item(), 2))
        print("[Test] ", prefix, round(top1.avg.item(), 2),' batch_time is ', batch_time.avg, ' data_time is ', data_time.avg)

    return losses.avg, top1.avg.to("cpu", torch.float).item()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def predict_softmax(predict_loader, model, device = 0):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in predict_loader:
            if torch.cuda.is_available():
                # images1 = Variable(images1).cuda()
                # images2 = Variable(images2).cuda()
                N,V,C,H,W = images1.size()
                images1 = Variable(images1).view(-1,C,H,W).to(device)
                images2 = Variable(images2).view(-1,C,H,W).to(device)
                # _, logits1 = model(images1)
                # _, logits2 = model(images2)
                if args.lossv:
                    _, logits1, _ = model(images1)
                    _, logits2, _ = model(images2)
                else:
                    _, logits1 = model(images1)
                    _, logits2 = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def target_transform(label):
    label = np.array(label, dtype=np.int32)
    target = torch.from_numpy(label).long()
    return target

def save_checkpoint(epoch, model, best_test_acc, optimizer, scheduler, filename):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_prec1': best_test_acc,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
    },  filename
    )
    print("saved ", filename)  

# Prepare train data loader
# original_train_data = kvDic['train_data']
# original_train_labels = kvDic['train_labels']

testdir = args.test_path
test_dataset = datasets.ImageFolder(
        testdir, 
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),)
print(test_dataset.class_to_idx)

original_train_data = []
original_train_labels = []
train_path = args.train_path
test_path = args.test_path
cleandata_path = args.cleandata_path
bgndata_path = args.bgndata_path
lndata_path = args.lndata_path

if args.num_classes == 10:
    classes_to_idx = {'airplane&aeroplane&plane': 0, 'bag&traveling_bag&travelling_bag&grip&suitcase': 1, 'bed': 2, 'car&auto&automobile&machine&motorcar': 3, 'file&file_cabinet&filing_cabinet': 4, 'knife&dagger&bayonet': 5, 'laptop&laptop computer': 6, 'rocket&projectile': 7, 'sofa&couch&lounge': 8, 'train&railroad_train': 9}
elif args.num_classes == 40:
    classes_to_idx = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
elif args.num_classes == 55:
    classes_to_idx = {'airplane': 0, 'ashcan': 1, 'bag': 2, 'basket': 3, 'bathtub': 4, 'bed': 5, 'bench': 6, 'bicycle': 7, 'birdhouse': 8, 'bookshelf': 9, 'bottle': 10, 'bowl': 11, 'bus': 12, 'cabinet': 13, 'camera&photograph_camera': 14, 'can&tin&tin_can': 15, 'cap': 16, 'car&auto&automobile&machine&motorcar': 17, 'chair': 18, 'clock': 19, 'computer_keyboard&keypad': 20, 'dishwasher&dish_washer&dishwashing_machine': 21, 'display&vedio_diaplay': 22, 'earphone&earpiece&headphone&phone': 23, 'faucet&spigot': 24, 'file&file_cabinet&filing_cabinet': 25, 'guitar': 26, 'helmet': 27, 'jar&vase': 28, 'knife&dagger&bayonet': 29, 'lamp&table lamp&floor lamp': 30, 'laptop&laptop computer': 31, 'loudspeaker&speaker&subwoofer&tweeter': 32, 'mailbox&letter box': 33, 'microphone&mike': 34, 'microwave&oven': 35, 'motorcycle&bike': 36, 'mug': 37, 'piano&pianoforte&forte_piano': 38, 'pillow': 39, 'pistol&handgun&side arm&shooting iron': 40, 'pot&flowerpot': 41, 'printer&printing_machine': 42, 'remote_control&remote': 43, 'rifle': 44, 'rocket&projectile': 45, 'skateboard': 46, 'sofa&couch&lounge': 47, 'stove': 48, 'table': 49, 'telephone&phone&telephone_set': 50, 'tower': 51, 'train&railroad_train': 52, 'vessel&watercraft': 53, 'washer&automatic_washer&washing_machine': 54}
elif args.num_classes == 20:
    classes_to_idx = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'chair': 4, 'cone': 5, 'cup': 6, 'curtain': 7, 'desk': 8, 'door': 9, 'dresser': 10, 'flower_pot': 11, 'glass_box': 12, 'guitar': 13, 'keyboard': 14, 'lamp': 15, 'laptop': 16, 'mantel': 17, 'monitor': 18, 'person': 19}
print(classes_to_idx)

for c in sorted(os.listdir(train_path)):
    for file in sorted(os.listdir(os.path.join(train_path, c))):
        original_train_data.append(os.path.join(c, file))
        original_train_labels.append(classes_to_idx[c])
original_train_data = np.array(original_train_data)
original_train_labels = np.array(original_train_labels)
print(len(original_train_labels))
length = int(len(original_train_labels) / args.num_views)
print(length)

train_labels = []
train_data = []
for i in range(length):
    grep = []
    for iview in range(args.num_views):
        grep.append(original_train_data[i * args.num_views + iview])
    train_labels.append(original_train_labels[i * args.num_views])
    train_data.append(grep)
print(len(train_data))
print(len(train_labels))

# shuffle_index = np.arange(len(original_train_labels), dtype=int)
# np.random.shuffle(shuffle_index)
# original_train_data = original_train_data[shuffle_index]
# original_train_labels = original_train_labels[shuffle_index]


original_test_data = []
original_test_labels = []
for c in sorted(os.listdir(test_path)):
    for file in sorted(os.listdir(os.path.join(test_path, c))):
        original_test_data.append(os.path.join(c, file))
        original_test_labels.append(classes_to_idx[c])
test_labels = []
test_data = []
length = int(len(original_test_labels) / args.num_views)
for i in range(length):
    grep = []
    for iview in range(args.num_views):
        grep.append(original_test_data[i * args.num_views + iview])
    test_labels.append(original_test_labels[i * args.num_views])
    test_data.append(grep)
print(len(test_data))
print(len(test_labels))


test_dataset = Clothing1M_Dataset_Multiview(test_data, test_labels, test_path, transform_test, target_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size * 2, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=False)

if cleandata_path != None:
    cleandata_data = []
    cleandata_labels = []
    for c in sorted(classes_to_idx.keys()):        
        i = 0
        for file in sorted(os.listdir(os.path.join(cleandata_path, c))):
            if i == 0:
                cleandata_data.append(os.path.join(c, file))
                cleandata_labels.append(classes_to_idx[c])
            i = (i + 1) % args.num_views
print(len(cleandata_data))
print(len(cleandata_labels))

if bgndata_path != None:
    bgndata_data = []
    bgndata_labels = []
    for c in sorted(classes_to_idx.keys()):
        i = 0
        for file in sorted(os.listdir(os.path.join(bgndata_path, c))):
            if i == 0:
                bgndata_data.append(os.path.join(c, file))
                bgndata_labels.append(classes_to_idx[c])
            i = (i + 1) % args.num_views
print(len(bgndata_data))
print(len(bgndata_labels))

if lndata_path != None:
    lndata_data = []
    lndata_labels = []
    for c in sorted(classes_to_idx.keys()):
        i = 0
        for file in sorted(os.listdir(os.path.join(lndata_path, c))):
            if i == 0:
                lndata_data.append(os.path.join(c, file))
                lndata_labels.append(classes_to_idx[c])
            i = (i + 1) % args.num_views
print(len(lndata_data))
print(len(lndata_labels))

# Prepare new data loader
nosie_len = int(len(train_labels) * args.data_percent)
whole_train_data = train_data[:nosie_len]
whole_train_labels = train_labels[:nosie_len]

train_dataset = Clothing1M_Dataset_Multiview(whole_train_data, whole_train_labels, train_path, transform, target_transform)
train_loader1 = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

model1 = create_model(args.pretrain)
optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
# scheduler1 = MultiStepLR(optimizer1, milestones=[30], gamma=0.5)
# print("lr 30 0.5")
# scheduler1 = MultiStepLR(optimizer1, milestones=[20,40], gamma=0.1)
# print("lr 20,40 0.1")
scheduler1 = MultiStepLR(optimizer1, milestones=[30, 60], gamma=0.5)
print("lr 30,60 0.5")
# scheduler1 = MultiStepLR(optimizer1, milestones=[50,80], gamma=0.1) # 55
# scheduler1 = MultiStepLR(optimizer1, milestones=[80,100,120], gamma=0.1)#40

# Loss function
train_nums = np.zeros(args.num_classes, dtype=int)
for item in whole_train_labels:
    train_nums[item] += 1
print("train categroy mean:", np.mean(train_nums, dtype=int), ", category:", train_nums, ", precent:", np.mean(train_nums) / train_nums)
# class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).to(device)
# ceriation1 = nn.CrossEntropyLoss(weight=class_weights).to(device)
ceriation1 = nn.CrossEntropyLoss().to(device)

best_val_acc = 0
best_test_acc = 0
filepath1_1 = args.model_dir + "/checkpoint1.pth"
filepath1_2 = args.model_dir + "/best_epoch1.pth"

# load resume
# pth
# args.start_epoch = 50
# if args.resume:
#     if os.path.isfile(args.resume):
#         print("=> loading checkpoint '{}'".format(args.resume))

#         checkpoint = torch.load(args.resume)
#         model1.load_state_dict(checkpoint)
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))
#     train_loader1, _, ceriation1 = update_trainloader(model1, whole_train_data, whole_train_labels)
# pth.tar
# args.resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/web40_multiview_fine_pes_splitdata_triplet/after_T1.pth.tar'
# args.resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/select/no_lr_5e-5/29_T1.pth.tar'
if args.resume != None and os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    # checkpoint = torch.load(args.resume, map_location={'cuda:0':'cuda:1'})
    checkpoint = torch.load(args.resume)
    # args.start_epoch = (checkpoint['epoch'] + 1) * 10    
    args.start_epoch = checkpoint['epoch']
    print("start epoch = ", args.start_epoch)
    best_test_acc = checkpoint['best_prec1']
    print("best_test_acc = ", best_test_acc)
    model1.load_state_dict(checkpoint['state_dict'])
    optimizer1.load_state_dict(checkpoint['optimizer'])
    scheduler1.load_state_dict(checkpoint['scheduler'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))

# initialize
cur_clean_trainloader= None
cur_edge_trainloader = None
edge_ceriation = None
clean_ceriation = None
if args.lossv:
    loss_view_cer = loss_view(args.beta)
    ceriation_all = Mixup(device, num_classes=args.num_classes, alpha=args.alpha, lossv = args.lossv, loss_view = loss_view_cer)
else:    
    ceriation_all = Mixup(device, num_classes=args.num_classes, alpha=args.alpha)
    loss_view_cer = None
# ceriation_all = Mixup(device, num_classes=args.num_classes, alpha=args.alpha)

print(getTime(), "Start train")
# train_third(model1, train_loader1, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
        
# _, test_acc = evaluate(model1, test_loader, ceriation1, " Test Acc:")
# _, test_acc = evaluate(model1, train_loader1, ceriation1, "Train Test Acc:")
if args.test_mode:
    sys.exit()
for epoch in range(args.start_epoch, args.num_epochs):
    #test
    # cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split(model1, whole_train_data, whole_train_labels)
    # train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
    # save_checkpoint(epoch,model1, best_test_acc, optimizer1, filename = 'test.pth.tar')
    # if args.checkdata:
    #         cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split(model1, whole_train_data, whole_train_labels, cleandata_data, cleandata_labels)
    # else:
    #     cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split(model1, whole_train_data, whole_train_labels)
    
    if epoch == args.T1:
        # print("no T2 ")
        # resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/web40_multiview_fine_pes_splitdata/after_pes.pth.tar'
        # resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/web40_multiview_savepth/30_pes.pth.tar'
        # resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/select/lr_20_10/after_pes.pth.tar'
        # print("=> loading checkpoint '{}'".format(resume))
        # checkpoint = torch.load(resume, map_location={'cuda:0':'cuda:1'})
        # # checkpoint = torch.load(resume)
        # model1.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(resume, checkpoint['epoch']))
        
        save_checkpoint(epoch, model1, best_test_acc, optimizer1,  scheduler1, filename = args.model_dir + '/after_T1.pth.tar')
        # model1 = noisy_refine(model1, train_loader1, ceriation1, args.T2)
        # save_checkpoint(epoch,model1, best_test_acc, optimizer1,  scheduler1, filename = args.model_dir + '/after_pes.pth.tar')
        '''
        if args.checkdata:
            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
        else:
            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
        '''
        if args.checkdata:
            if args.select_type == 0:
                # print("updating trainloader fine&PES and split dataloader...")
                if args.mixtype == 0 or args.mixtype == 3:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                elif args.mixtype == 1:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                elif args.mixtype == 2:
                    None
            elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                cur_clean_trainloader, clean_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
        else:
            if args.select_type == 0:
                if args.mixtype == 0 or args.mixtype == 3:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                elif args.mixtype == 1:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                elif args.mixtype == 2:
                    None
            elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                cur_clean_trainloader, clean_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
    else:
        # if epoch < args.T1 or epoch > args.T1:
        if epoch < args.T1:
            print("[Train]  [Epoch] ", str(epoch))
            # if args.mixtype == 2:
            #     train(model1, train_loader1, ceriation_all, optimizer1)
            # else:
            train(model1, train_loader1, ceriation1, optimizer1, loss_view_cer = loss_view_cer)
            scheduler1.step()
        
        else: #>args.T1
            print("[Train]  [Epoch] ", str(epoch))
            # train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
            # train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
            # train_third(model1, train_loader1, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
            if args.select_type == 0:
                if args.mixtype == 0:
                    train_third(model1, train_loader1, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1, loss_view_cer = loss_view_cer)
                elif args.mixtype == 1:
                    train_third(model1, tmp_trainloader, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1, loss_view_cer = loss_view_cer)
                elif args.mixtype == 3:
                    train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1, loss_view_cer = loss_view_cer)
                elif args.mixtype == 2:
                    train(model1, train_loader1, ceriation_all, optimizer1, isMixup = True, loss_view_cer = loss_view_cer)
            elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                train(model1, cur_clean_trainloader, clean_ceriation, optimizer1, loss_view_cer = loss_view_cer)

            scheduler1.step()
             # check_selectdata(model1,  whole_train_data, whole_train_labels, cleandata_data, cleandata_labels)
            if (epoch + 1 - args.T1) % args.every == 0 and (args.select_type != 0 or args.mixtype != 2):
                if args.checkdata:
                    if args.select_type == 0:
                        # print("updating trainloader fine&PES and split dataloader...")
                        if args.mixtype == 0 or args.mixtype == 3:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                        elif args.mixtype == 1:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                        elif args.mixtype == 2:
                            None
                    elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                        cur_clean_trainloader, clean_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                else:
                    if args.select_type == 0:
                        if args.mixtype == 0 or args.mixtype == 3:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                        elif args.mixtype == 1:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                        elif args.mixtype == 2:
                            None
                    elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                        cur_clean_trainloader, clean_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
        
        # _, train_acc = evaluate(model1, train_loader1, ceriation1, "Epoch " + str(epoch) + ", Train Acc:")
        _, test_acc = evaluate(model1, test_loader, ceriation1, "Epoch " + str(epoch) + ", Test Acc:")
        # torch.save(model1.state_dict(), filepath1_1)
        save_checkpoint(epoch,model1, best_test_acc, optimizer1, scheduler1, filename = filepath1_1)
        # if((epoch + 1) % 10 == 0):
        #     save_checkpoint(epoch,model1, best_test_acc, optimizer1, scheduler1, filename = args.model_dir + '/'+ str(epoch) + '_T1.pth.tar')
        
        if best_test_acc < test_acc:
            torch.save(model1.state_dict(), filepath1_2)
            best_test_acc = test_acc
        print("Model1 Best Test Acc:", best_test_acc)        
       
'''
for epoch in range(args.start_epoch, args.num_epochs):
    #test
    # cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split(model1, whole_train_data, whole_train_labels)
    # train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
    # save_checkpoint(epoch,model1, best_test_acc, optimizer1, filename = 'test.pth.tar')
    # if args.checkdata:
    #         cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split(model1, whole_train_data, whole_train_labels, cleandata_data, cleandata_labels)
    # else:
    #     cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split(model1, whole_train_data, whole_train_labels)
    
    if epoch == args.T1:
        # print("no T2 ")
        # resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/web40_multiview_fine_pes_splitdata/after_pes.pth.tar'
        # resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/web40_multiview_savepth/30_pes.pth.tar'
        # resume = '/home/gq/projects/cwb/noise_methods/sample_selection/PES-main/save_models/webnet40/select/lr_20_10/after_pes.pth.tar'
        # print("=> loading checkpoint '{}'".format(resume))
        # checkpoint = torch.load(resume, map_location={'cuda:0':'cuda:1'})
        # # checkpoint = torch.load(resume)
        # model1.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})"
        #         .format(resume, checkpoint['epoch']))
        
        save_checkpoint(epoch, model1, best_test_acc, optimizer1,  scheduler1, filename = args.model_dir + '/after_T1.pth.tar')
        # model1 = noisy_refine(model1, train_loader1, ceriation1, args.T2)
        # save_checkpoint(epoch,model1, best_test_acc, optimizer1,  scheduler1, filename = args.model_dir + '/after_pes.pth.tar')

        # if args.checkdata:
        #     cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
        # else:
        #     cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)

        if args.checkdata:
            if args.select_type == 0:
                print("updating trainloader fine&PES and split dataloader...")
                if args.mixtype == 0 or args.mixtype == 3:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                elif args.mixtype == 1:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                elif args.mixtype == 2:
                    None
            elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                cur_clean_trainloader, clean_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
        else:
            if args.select_type == 0:
                if args.mixtype == 0 or args.mixtype == 3:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                elif args.mixtype == 1:
                    cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                elif args.mixtype == 2:
                    None
            elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                cur_clean_trainloader, clean_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
    else:
        # if epoch < args.T1 or epoch > args.T1:
        if epoch < args.T1:
            print("[Train]  [Epoch] ", str(epoch))
            # if args.mixtype == 2:
            #     train(model1, train_loader1, ceriation_all, optimizer1)
            # else:
            train(model1, train_loader1, ceriation1, optimizer1)
            scheduler1.step()
        
        else: #>args.T1
            print("[Train]  [Epoch] ", str(epoch))
            # train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
            # train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
            # train_third(model1, train_loader1, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
            if args.select_type == 0:
                if args.mixtype == 0:
                    train_third(model1, train_loader1, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
                elif args.mixtype == 1:
                    train_third(model1, tmp_trainloader, ceriation_all, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
                elif args.mixtype == 3:
                    train_second(model1, cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, optimizer1)
                elif args.mixtype == 2:
                    train(model1, train_loader1, ceriation_all, optimizer1, isMixup = True)
            elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                train(model1, cur_clean_trainloader, clean_ceriation, optimizer1)

            scheduler1.step()
             # check_selectdata(model1,  whole_train_data, whole_train_labels, cleandata_data, cleandata_labels)
            if (epoch + 1 - args.T1) % args.every == 0 and (args.select_type != 0 or args.mixtype != 2):
                if args.checkdata:
                    if args.select_type == 0:
                        print("updating trainloader fine&PES and split dataloader...")
                        if args.mixtype == 0 or args.mixtype == 3:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                        elif args.mixtype == 1:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                        elif args.mixtype == 2:
                            None
                    elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                        cur_clean_trainloader, clean_ceriation = check_selectdata_fine_pes_split_half(model1, whole_train_data, whole_train_labels, cleandata_data, bgndata_data, cleandata_labels, bgndata_labels)
                else:
                    if args.select_type == 0:
                        if args.mixtype == 0 or args.mixtype == 3:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                        elif args.mixtype == 1:
                            cur_clean_trainloader, clean_ceriation, cur_edge_trainloader, edge_ceriation, tmp_trainloader = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
                        elif args.mixtype == 2:
                            None
                    elif args.select_type == 1 or args.select_type == 2 or args.select_type == 3:
                        cur_clean_trainloader, clean_ceriation = update_trainloader_fine_pes_split_half(model1, whole_train_data, whole_train_labels)
        
        # _, train_acc = evaluate(model1, train_loader1, ceriation1, "Epoch " + str(epoch) + ", Train Acc:")
        _, test_acc = evaluate(model1, test_loader, ceriation1, "Epoch " + str(epoch) + ", Test Acc:")
        # torch.save(model1.state_dict(), filepath1_1)
        save_checkpoint(epoch,model1, best_test_acc, optimizer1, scheduler1, filename = filepath1_1)
        # if((epoch + 1) % 10 == 0):
        #     save_checkpoint(epoch,model1, best_test_acc, optimizer1, scheduler1, filename = args.model_dir + '/'+ str(epoch) + '_T1.pth.tar')
        
        if best_test_acc < test_acc:
            torch.save(model1.state_dict(), filepath1_2)
            best_test_acc = test_acc
        print("Model1 Best Test Acc:", best_test_acc)        
'''     

print(getTime(), "Last----Model1 Best Test Acc:", best_test_acc)