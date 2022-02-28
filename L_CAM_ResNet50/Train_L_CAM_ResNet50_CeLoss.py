import sys
sys.path.append('../')
import torch
import argparse
import os
import time
import shutil
import json
import datetime
import L_CAM_ResNet50.my_optim as my_optim
from torch.autograd import Variable
from utils import AverageMeter
from utils import Metrics
from utils.LoadData import data_loader
from utils.Restore import restore
from utils.Restore import restore
from models import ResNet50_L_CAM_ImgA, ResNet50_L_CAM_FmA
import torch.nn.functional as F
import torch.nn as nn

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:',ROOT_DIR)
IMG_DIR  = r'/m2/ILSVRC2012_img_train'

# Static paths
train_list = os.path.join(ROOT_DIR,'datalist', 'ILSVRC', 'ResNet50_train.txt')
test_list = os.path.join(ROOT_DIR,'datalist','ILSVRC', 'Evaluation_2000.txt')
Snapshot_dir = os.path.join(ROOT_DIR,'snapshots', 'ResNet50_L_CAM_ImgA')

# Default parameters
EPOCH = 26
Batch_size = 64
disp_interval = 40
num_workers = 1
num_classes = 1000
dataset = 'imagenet'
LR = 0.0001
restore_from= r'/home/xiaolin/.torch/models/vgg16-397923af.pth'


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet50_Att_CS')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR,help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default=IMG_DIR, help='Directory of training images')
    parser.add_argument("--train_list", type=str, default=train_list)
    parser.add_argument("--test_list", type=str, default=test_list)
    parser.add_argument("--batch_size", type=int, default=Batch_size)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument("--num_classes", type=int, default=num_classes)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--arch", type=str,default='ResNet50_L_CAM_ImgA')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=num_workers)
    parser.add_argument("--disp_interval", type=int, default=disp_interval)
    parser.add_argument("--snapshot_dir", type=str, default=Snapshot_dir)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='False')
    parser.add_argument("--restore_from", type=str, default=restore_from)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = eval(args.arch).model()                                  
    model.cuda()
    lr = args.lr
    optimizer = my_optim.get_finetune_optimizer(args, model)

    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return  model, optimizer

def train(args):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model, optimizer= get_model(args)
    model.cuda()        
    model.train()   
      
    for param in model.conv1.parameters():  
        param.requires_grad = False
    for param in model.bn1.parameters():  
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False        
    for param in model.fc.parameters():
        param.requires_grad = False         


    def _freeze_norm_stats(net):
        try:
            for m in net.modules():              
                if isinstance(m, nn.BatchNorm2d):                    
                    m.eval()  
        except ValueError:  
            print("eror with instancenorm")
            return
    model = model.cuda()
    model.train()
    
    train_loader, _ = data_loader(args)

    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('#epoch,loss,pred@1,pred@5\n')


    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    end = time.time()
    max_iter = total_epoch*len(train_loader)
    print('Max iter:', max_iter)
    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        top1.reset()
        top5.reset()
        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)

        if res:
            for g in optimizer.param_groups:
                out_str = 'Epoch:%d, %f\n'%(current_epoch, g['lr'])
        it = 0
        steps_per_epoch = len(train_loader)
        for idx, dat in enumerate(train_loader):
            model.conv1.apply(_freeze_norm_stats)
            model.bn1.apply(_freeze_norm_stats)
            model.relu.apply(_freeze_norm_stats)
            model.maxpool.apply(_freeze_norm_stats)
            model.layer1.apply(_freeze_norm_stats)
            model.layer2.apply(_freeze_norm_stats)
            model.layer3.apply(_freeze_norm_stats)
            model.layer4.apply(_freeze_norm_stats)
            model.avgpool.apply(_freeze_norm_stats)
            model.fc.apply(_freeze_norm_stats)

            
            it = it + 1 
                
            img_path , img, label = dat

            global_counter += 1
            img, label = img.cuda(), label.cuda()
            img_var, label_var = Variable(img), Variable(label)

            logits = model(img_var,  label_var)
            
            loss_val, = model.get_loss(logits, label_var)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            logits1 = torch.squeeze(logits[0])
            prec1_1, prec5_1 = Metrics.accuracy(logits1.data, label.long(), topk=(1,5))
            top1.update(prec1_1[0], img.size()[0])
            top5.update(prec5_1[0], img.size()[0])

            losses.update(loss_val.data, img.size()[0])
            batch_time.update(time.time() - end)

            end = time.time()
            if global_counter % 1000 == 0:
                losses.reset()
                top1.reset()
                top5.reset()

            if global_counter % args.disp_interval == 0:
                # Calculate ETA
                eta_seconds = ((total_epoch - current_epoch)*steps_per_epoch + (steps_per_epoch - idx))*batch_time.avg
                eta_str = (datetime.timedelta(seconds=int(eta_seconds)))
                eta_seconds_epoch = steps_per_epoch*batch_time.avg
                eta_str_epoch = (datetime.timedelta(seconds=int(eta_seconds_epoch)))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ETA {eta_str}({eta_str_epoch})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                     .format(
                    current_epoch, global_counter%len(train_loader), len(train_loader), batch_time=batch_time,
                    eta_str=eta_str, eta_str_epoch = eta_str_epoch, loss=losses, top1=top1, top5=top5))                               
                
        if current_epoch % 1 == 0:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'arch': 'resnet',
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d_glo_step_%d.pth.tar'
                                      %(args.dataset, current_epoch,global_counter))

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            fw.write('%d,%.4f,%.3f,%.3f\n'%(current_epoch, losses.avg, top1.avg, top5.avg))

        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
