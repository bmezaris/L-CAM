#!/bin/sh

cd ../exper_VGG16_Attention/
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_0' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_1' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_2' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_3' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_4' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_5' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_6' \
	--percentage=0 \
	
CUDA_VISIBLE_DEVICES=0 python VGG16_Att_CS_AD_IC.py \
	--snapshot_dir='/m2/gkartzoni/Aux-DCNN/snapshots/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2/VGG16_Att_CS_sigmoidSaliency_fmxsal_CL_v2_7' \
	--percentage=0 \
