#!/bin/sh

cd ../L_CAM_VGG16/

CUDA_VISIBLE_DEVICES=0 python Evaluation_L_CAM_VGG16.py \
	--img_dir='/m2/ILSVRC2012_img_val' \
	--snapshot_dir='/m2/gkartzoni/L-CAM/snapshots/VGG16_L_CAM_Img' \
	--arch='VGG16_L_CAM_Img' \
	--percentage=0 \

