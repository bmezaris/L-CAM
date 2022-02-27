#!/bin/sh

cd ../exper_ResNet50_Attention/
CUDA_VISIBLE_DEVICES=0 python ResNet50_Att_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/ResNet50_Att_CS_sigmoidSaliency_imgXsal_CEL_v2' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python ResNet50_Att_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/ResNet50_Att_CS_sigmoidSaliency_imgXsal_CEL_v2' \
	--percentage=0.5 \
	
CUDA_VISIBLE_DEVICES=0 python ResNet50_Att_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/ResNet50_Att_CS_sigmoidSaliency_imgXsal_CEL_v2' \
	--percentage=0.7 \
	
CUDA_VISIBLE_DEVICES=0 python ResNet50_Att_ResNet18_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/ResNet50_Att_CS_sigmoidSaliency_imgXsal_CEL_v2' \
	--percentage=0.85 \
