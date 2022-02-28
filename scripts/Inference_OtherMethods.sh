#!/bin/sh

cd ../Inference_OtherMethods/
CUDA_VISIBLE_DEVICES=0 python VGG16_gradcam.py \
	--percentage=0 \
	--img_dir='/m2/ILSVRC2012_img_val' \

