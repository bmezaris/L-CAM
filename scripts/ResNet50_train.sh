#!/bin/sh

cd ../L_CAM_ResNet50/

CUDA_VISIBLE_DEVICES=0 python Train_L_CAM_ResNet50.py \
	--img_dir='/m2/ILSVRC2012_img_train' \
	--snapshot_dir='/m2/gkartzoni/L-CAM/snapshots/ResNet50_L_CAM_Img' \
	--arch='ResNet50_L_CAM_Img' \







