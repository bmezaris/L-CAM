#!/bin/sh

cd ../exper_VGG16_Attention/
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_ChangeLPar' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_ChangeLPar' \
	--percentage=0.5 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_ChangeLPar' \
	--percentage=0.7 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_ChangeLPar' \
	--percentage=0.85 \
