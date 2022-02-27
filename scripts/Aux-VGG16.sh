#!/bin/sh

cd ../exper_VGG16_Aux/

CUDA_VISIBLE_DEVICES=0 python VGG16_aux_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/Aux_DCNN_VGG16_betterLoss_ReluDivMax' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_aux_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/Aux_DCNN_VGG16_betterLoss_ReluDivMax' \
	--percentage=0.5 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_aux_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/Aux_DCNN_VGG16_betterLoss_ReluDivMax' \
	--percentage=0.7 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_aux_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/Aux_DCNN_VGG16_betterLoss_ReluDivMax' \
	--percentage=0.85 \
	