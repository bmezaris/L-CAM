#!/bin/sh

cd ../exper_OtherMethods/

CUDA_VISIBLE_DEVICES=0 python VGG16_AD_IC.py \
	--percentage=0 \
    
CUDA_VISIBLE_DEVICES=0 python ResNet50_AD_IC.py \
	--percentage=0.5 \
    
CUDA_VISIBLE_DEVICES=0 python ResNet50_AD_IC.py \
	--percentage=0.7 \
    
CUDA_VISIBLE_DEVICES=0 python ResNet50_AD_IC.py \
	--percentage=0.85 \
