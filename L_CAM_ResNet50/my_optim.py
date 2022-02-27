import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list =[]
    for name,value in model.named_parameters():
        if '-'   in  name:
            print('Bigger weights',name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)
    weight_decay = 0.00001
    opt = optim.SGD([{'params': weight_list, 'lr':lr},
                      {'params':bias_list, 'lr':lr*2},
                      {'params':last_weight_list, 'lr':lr*100},
                      {'params': last_bias_list, 'lr':lr*200}], momentum=0.9, weight_decay=0.0005, nesterov=True)
    return opt

def lr_poly(base_lr, iter,max_iter,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)

def get_optimizer(args, model):
    lr = args.lr
    opt = optim.SGD(params=[para for name, para in model.named_parameters() if 'features' not in name], lr=lr, momentum=0.9, weight_decay=0.0001)
    return opt


def reduce_lr(args, optimizer, epoch, factor=0.95):
    change_points = [1,2,3,4,5,6,7,8,9,10,11,12]
    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            #g['lr'] = g['lr']
            g['lr'] = g['lr']*factor
            print(epoch, g['lr'])
        return True


