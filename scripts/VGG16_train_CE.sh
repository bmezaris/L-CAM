#!/bin/sh

cd ../L_CAM_VGG16/

CUDA_VISIBLE_DEVICES=0 python Train_L_CAM_VGG16_CeLoss.py \
	--img_dir='/m2/ILSVRC2012_img_train' \
	--snapshot_dir='/m2/gkartzoni/L-CAM/snapshots/VGG16_L_CAM_ImgA' \
	--arch='VGG16_L_CAM_ImgA' \


