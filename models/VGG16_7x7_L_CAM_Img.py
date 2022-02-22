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

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = self.get_features()
        self.features = nn.Sequential(*list(features)[:31])
        classifier = self.get_classifier()
        self.classifier = nn.Sequential(*list(classifier)[:7])
        self.adapPool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.attnM= AttentionMechanismM(in_features=512)
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.max_pool =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.area_loss_coef = 2
        self.smoothness_loss_coef = 0.01
        self.ce_coef = 1.5
        self.area_loss_power = 0.3
    def get_features(self):
        vgg = models.vgg16(pretrained=True)
        return vgg.features 

    def get_classifier(self):
        vgg = models.vgg16(pretrained=True)
        return vgg.classifier   
    
    def forward(self, x, label,isTrain=True):
        # FMs
        x_norm =  Metrics.normalize(x)   
        l = self.features(x_norm)
        # Attention
        a1,c1 = self.attnM(l)
        self.a = a1
        self.c = c1

        if isTrain==True:
            N, C, W, H = a1.size()
            temp = torch.arange(0,N).cuda()*1000 #a vector [0,...,1000*Ν]
            t = label.long() + temp #multiply label(etc [9,2,5,....,9]) with vector [0,...,Ν] find index
            a1 = torch.reshape(a1, (N*C,W,H)) 
            a2 = a1[t.long(),:,:] # #Take indeces
            a2 = torch.unsqueeze(a2, 1)
            self.a2 = a2
            a2 = F.interpolate(a2, size=(224,224), mode='bilinear')
            x_masked = torch.mul(a2, x) 
            x_norm =  Metrics.normalize(x_masked) 
        # classifier
        y = self.features(x_norm)
        y = self.adapPool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return [y, ]
        
    def get_loss(self, logits, gt_labels,masks):
        gt = gt_labels.long()
        cross_entr = self.loss_cross_entropy(logits[0], gt)*self.ce_coef 
        area_loss = self.area_loss_coef*self.area_loss(masks) 
        varation_loss = self.smoothness_loss_coef*self.smoothness_loss(masks)
        loss = cross_entr  + area_loss + varation_loss
        return [loss, cross_entr, area_loss, varation_loss]
        
    def area_loss(self, masks):
        if self.area_loss_power != 1:
            masks = (masks+0.0005)**self.area_loss_power # prevent nan (derivative of sqrt at 0 is inf)
        return torch.mean(masks)
  
    def smoothness_loss(self,masks, power=2, border_penalty=0.3):
        x_loss = torch.sum((torch.abs(masks[:,:,1:,:] - masks[:,:,:-1,:]))**power)
        y_loss = torch.sum((torch.abs(masks[:,:,:,1:] - masks[:,:,:,:-1]))**power)
        if border_penalty>0:
            border = float(border_penalty)*torch.sum(masks[:,:,-1,:]**power + masks[:,:,0,:]**power + masks[:,:,:,-1]**power + masks[:,:,:,0]**power)
        else:
            border = 0.
        return (x_loss + y_loss + border) / float(power * masks.size(0))  # watch out, normalised by the batch size!

    
    def get_c(self, gt_label):
        map1 = self.c
        map1 = map1[:,gt_label,:,:]
        return [map1,]

    def get_a(self, gt_label):
        map1 = self.a
        map1 = map1[:,gt_label,:,:]
        return [map1,]
   

def model(pretrained=True, **kwargs):
    model = VGG16()     
    return model

# from torchsummary import summary
# vgg = models.vgg16(pretrained=True)
# features = vgg.features 
# net = nn.Sequential(*list(features)[:30])
# print(net)
# net = net.cuda()
# summary(net, input_size=(3, 224, 224))
# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-12].data.cpu().numpy())
# print(weight_softmax.shape)
# print(net)


