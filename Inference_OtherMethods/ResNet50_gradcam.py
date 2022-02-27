import sys
sys.path.append('../')
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import time
import numpy as np
from torchvision import models, transforms
import torch.nn.functional as F
from utils import AverageMeter
from utils import Metrics
from utils.LoadData import data_loader
from utils.cam import GradCAM

os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:',ROOT_DIR)
IMG_DIR  = r'/m2/ILSVRC2012_img_val'

# Static paths
train_list = os.path.join(ROOT_DIR,'datalist', 'ILSVRC', 'VGG16_train.txt')
test_list = os.path.join(ROOT_DIR,'datalist','ILSVRC', 'Evaluation_2000.txt')

percent = 0
def get_arguments():
    parser = argparse.ArgumentParser(description='-')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--train_list", type=str, default=train_list)
    parser.add_argument("--test_list", type=str, default=test_list)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--arch", type=str,default='-')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=40)
    parser.add_argument("--snapshot_dir", type=str, default='/m2/gkartzoni/L-CAM/snapshots/temp')
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default=r'/home/xiaolin/.torch/models/vgg16-397923af.pth')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--percentage", type=float, default=percent)
    return parser.parse_args()



current_epoch=0
model=None
args = get_arguments()

top1 = AverageMeter()
top5 = AverageMeter()
top1.reset()
top5.reset()

top1_ = AverageMeter()
top5_ = AverageMeter()
top1_.reset()
top5_.reset()

def get_model(args):
    model = models.resnet50(pretrained=True).eval()
    model.cuda()
    return  model

if model is None:
    model = get_model(args)


train_loader, val_loader = data_loader(args, test_path=True)

global_counter = 0
prob = None
gt = None

y_mask_image = []
y_image = []

model.cuda()


target_layer = model.layer4[2]
wrapped_model = GradCAM(model, target_layer)
k = 0
time_spend = 0
import matplotlib.image as mpimg 
  

for idx, dat  in tqdm(enumerate(val_loader)):
    img_path, img, label_in = dat
    im = img
    global_counter += 1
    label = label_in
    img, label = img.cuda(), label.cuda()
    
    now = time.time()
    with torch.no_grad():
        img_n = Metrics.normalize(img)
        logits = model(img_n)
    logits0 = logits
    logits0 = F.softmax(logits0, dim=1)

    prec1_1, prec5_1 = Metrics.accuracy(logits0.cpu().data, label_in.long(), topk=(1,5))
    y = logits0.cpu().data.numpy()
    class_1 = logits0.max(1)[-1]  #

    index_gt_y = class_1.long().cpu().data.numpy()  #
    
    Y_i_c = logits0.max(1)[0].item()
    y_image.append(Y_i_c)
    
    top1.update(prec1_1[0], img.size()[0])
    top5.update(prec5_1[0], img.size()[0])

    cam_map,_ = wrapped_model(img_n,index_gt_y)
    cam_map = F.interpolate(cam_map, size=(224,224), mode='bilinear', align_corners=False)
    cam_map = Metrics.drop_Npercent(cam_map,args.percentage)

    
    cam_map = cam_map.cuda() 
    image = img
    mask_image = cam_map * image   
       
    mask_image = mask_image.cuda()
    with torch.no_grad():
        img_n = Metrics.normalize(mask_image)
        logits = model(img_n)
    logits0 = logits
    logits0 = F.softmax(logits0, dim=1)


    prec1_1, prec5_1 = Metrics.accuracy(logits0.cpu().data, label_in.long(), topk=(1,5))
    y = logits0.cpu().data.numpy()
    
    Y_i_c_ = logits0[:,index_gt_y][0].item()
    y_mask_image.append(Y_i_c_)
    
    top1_.update(prec1_1[0], img.size()[0])
    top5_.update(prec5_1[0], img.size()[0])    

y_image = np.array(y_image)
y_mask_image = np.array(y_mask_image)


print('Average Drop is: ', Metrics.AD(y_image,y_mask_image))
print('Increase in confidence is: ', Metrics.IC(y_image,y_mask_image))
print('Top1:', top1.avg, 'Top5:',top5.avg)
print('Top1:', top1_.avg, 'Top5:',top5_.avg)
