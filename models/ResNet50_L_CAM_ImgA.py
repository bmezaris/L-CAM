import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16
import torchvision.models as models
import sys
sys.path.append('../')
from torchvision import models, transforms
from utils import Metrics

class AttentionMechanismM(nn.Module):
    def __init__(self, in_features):
        super(AttentionMechanismM, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1000, kernel_size=1, padding=0, bias=True)
    def forward(self, l):
        N, C, W, H = l.size()
        c = self.op(l) # batch_sizex1xWxH
        a = torch.sigmoid(c)
        return  a,c
    
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = nn.Sequential(*list(resnet.layer1))
        self.layer2 = nn.Sequential(*list(resnet.layer2))
        self.layer3 = nn.Sequential(*list(resnet.layer3))
        self.layer4 = nn.Sequential(*list(resnet.layer4))
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.attnM= AttentionMechanismM(in_features=2048)
        self.loss_cross_entropy = nn.CrossEntropyLoss()
  

    def forward(self, x, label = None, train = True):
        # FMs
        
        x_norm =  Metrics.normalize(x)   
        l = self.conv1(x)
        l =self.bn1(l)
        l = self.relu(l)
        l =self.maxpool(l) 
        l = self.layer1(l) 
        l = self.layer2(l)
        l = self.layer3(l) 
        l = self.layer4(l) 
        a1,c1 = self.attnM(l)
        self.a = a1
        self.c = c1
        if train==True:
           # print('train mode')
            N, C, W, H = a1.size()
            temp = torch.arange(0,N).cuda()*1000 #a vector [0,...,1000*Ν]
            t = label.long() + temp #multiply label(etc [9,2,5,....,9]) with vector [0,...,Ν] find index
            a1 = torch.reshape(a1, (N*C,W,H)) 
            a2 = a1[t.long(),:,:] # #Take indeces
            a2 = torch.unsqueeze(a2, 1)
            a2 = F.interpolate(a2, size=(224,224), mode='bilinear')
            x_masked = torch.mul(a2, x) 
            x_norm =  Metrics.normalize(x_masked)          
            
        y = self.conv1(x_norm)
        y =self.bn1(y)
        y = self.relu(y)
        y =self.maxpool(y) 
        y = self.layer1(y) 
        y = self.layer2(y)
        y = self.layer3(y) 
        y = self.layer4(y) 
        y = self.avgpool(y) 
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return [y, ]
        

    def get_loss(self, logits, gt_labels):
        gt = gt_labels.long()
        cross_entr = self.loss_cross_entropy(logits[0], gt)
        loss = cross_entr  
        return [loss]
        
        
    def get_c(self, gt_label):
        map1 = self.c
        map1 = map1[:,gt_label,:,:]
        return [map1,]
    
    def get_a(self, gt_label):
        map1 = self.a
        map1 = map1[:,gt_label,:,:]
        return [map1,]


def model(pretrained=True, **kwargs):
    model = Resnet50()       
    return model



