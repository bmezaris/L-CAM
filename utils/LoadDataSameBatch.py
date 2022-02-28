# from torchvision import transforms
from .transforms import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as my_dataset, dataset_with_mask
import torchvision
import torch
import numpy as np

def data_loader(args, test_path=False, segmentation=False):

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)


    tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])
    
    
    
    tsfm_val = transforms.Compose([
                                     transforms.Resize(input_size),  #356
                                     transforms.CenterCrop(crop_size), #321
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])


    img_train = my_dataset(args.train_list, root_dir=args.img_dir,
                           transform=tsfm_train, with_path=True)

    img_test = my_dataset(args.test_list, root_dir=args.img_dir,
                          transform=tsfm_val, with_path=test_path)

    class_inds = [torch.where(img_train.targets == class_idx)[0]
              for class_idx in img_train.class_to_idx.values()]


    train_dataloaders = [
    DataLoader(
        dataset=Subset(img_train, inds),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers)
    for inds in class_inds]


    val_dataloaders = [
    DataLoader(
        dataset=Subset(img_test, inds),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers)
    for inds in class_inds]


    #train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_dataloaders, val_dataloaders


