# L-CAM

This repository hosts the code and data lists for our two learning-based eXplainable AI (XAI) methods called L-CAM-Fm and L-CAM-Img, for deep convolutional neural networks (DCNN) image classifiers. Our methods receive as input an image and a class label and produce as output the image regions that the DCNN has focused on in order to infer this class. Both methods use an attention mechanism (AM), trained  end-to-end  along  with the original (frozen) DCNN, to derive class activation maps (CAMs) from the last convolutional layer’s feature maps (FMs). During training, CAMs are applied to the FMs (L-CAM-Fm) or the input image (L-CAM-Img), forcing the AM to learn the image regions explaining the DCNN’s outcome. Two widely-used evaluation metrics, Increase in Confidence (IC) and Average Drop (AD), are used for evaluation.
- This repository contains the code for training L-CAM-Fm and L-CAM-Img, using VGG-16 or ResNet-50 as the pre-trained backbone network along with the Attention Mechanism and our selected loss function. There is also code to train the above networks with the conventional cross-entropy loss. The models files with the model architecture are named as following:
First the name of the backbone (ResNet50 or VGG-16) and then the method's name (L-CAM-Fm or L-CAM-Img). If the model uses the cross-entropy loss (instead of our proposed loss function) there is also an A character at the end of the name, e.g. ResNet50_L_CAM_ImgA.py. There is also a variation L-CAM-Img with VGG-16 backbone where the AM's input is the 7×7 FMs after the last max pooling layer of VGG-16, in contrast to all the others models that use the last convolutional layer of the backbone. This model is named VGG16_7x7_L_CAM_Img.py.  
- Instead of training, the user can also download the pre-trained models for L-CAM-Fm and L-CAM-Img (again using VGG-16 or ResNet-50 as the backbone network along with the Attention Mechanism and our selected loss function [here](https://drive.google.com/drive/folders/1QiwB3iEobEPnSB9NRSsmDaUAuBMiPdz2?usp=sharing). The pre-trained models are named in the same way as the models files (as explained in the previous paragraph). 
- There is also code for evaluating our method according to two widely used evaluation metrics for DCNN explainability, Increase in Confidence (IC) and Average Drop (AD).  In the same script, Top-1 and Top-5 accuracy is as well calculated.
- Furthermore, there is the code to evaluate the methods that are used in our paper for comparisons with L-CAM-Fm and L-CAM-Img.
- In [L-CAM/datalist/ILSVRC](https://github.com/bmezaris/L-CAM/tree/main/datalist/ILSVRC) there are text files with annotations for training VGG-16 and ResNet-50 (VGG-16_train.txt, ResNet50_train.txt) and text files with annotations for 2000 randomly selected images to be used at the evaluation stage (Evaluation_2000.txt) for the L-CAM methods.
- The ImageNet1K dataset images should be downloaded by the user manually.
- This project is implemented and tested in python 3.6 and PyTorch 1.9.

## Basic Code requirements
- PyTorch
- cv2
- scikit-learn

## Visual examples and comparison of results 
![alt text](https://github.com/bmezaris/L-CAM/blob/main/images/superimposed.png)

## Data preparation
Download [here](https://image-net.org/) the training and evaluation images for ImageNet1K dataset, then extract folders and sub-folders and place the extracted folders (ILSVRC2012_img_train, ILSVRC2012_img_val) in the dataset/ILSVRC2012_img_train and dataset/ILSVRC2012_img_val folders. The folder structure of the image files should look as below:
```
dataset
    └── ILSVRC2012_img_train
        └── n01440764
            ├── n01440764_10026.JPEG
            ├── n01440764_10027.JPEG
            └── ...
        └── n01443537
        └── ...
    └── ILSVRC2012_img_val
        ├── ILSVRC2012_val_00000001.JPEG
        ├── ILSVRC2012_val_00000002.JPEG
        └── ...
```

## Install
- Clone this repository
~~~
git clone https://github.com/bmezaris/L-CAM
~~~
- Go to the locally saved repository path:
~~~
cd L-CAM
~~~
- Create the snapshots folder to save the trained models:
~~~
mkdir snapshots
~~~

## Training

- To train from scratch VGG-16 or ResNet-50, run for the VGG-16 backbone with the selected loss function:
~~~
cd scripts
sh VGG16_train.sh 
~~~
**OR**, for the VGG-16 backbone with cross-entropy loss:
~~~
cd scripts
sh VGG16_train_CE.sh 
~~~
**OR**, for the ResNet-50 backbone with the selected loss function:
~~~
cd scripts
sh ResNet50_train.sh
~~~
**OR**, for the ResNet-50 backbone with cross-entropy loss:
~~~
cd scripts
sh ResNet50_train_CE.sh 
~~~
Before running any of the .sh files, set the img_dir, snapshot_dir and arch parameters inside the .sh file. For the *_CE.sh files the arch parameter must be set only with model file's names (*/L-CAM/models) with the A character at the end, for all the other .sh files the arch parameter must be set with file's names (*/L-CAM/models) without the A character at the end. The produced model will be saved in the snapshots folder. 

## Evaluation of L-CAM-Fm and L-CAM-Img
- To evaluate the model, download the pretrained models that are available in this [GoogleDrive](https://drive.google.com/drive/folders/1QiwB3iEobEPnSB9NRSsmDaUAuBMiPdz2?usp=sharing), and place the downloaded folders (VGG16_L_CAM_Img, VGG16_L_CAM_Fm, VGG16_7x7_L_CAM_Img, ResNet50_L_CAM_Fm, ResNet50_L_CAM_Img) in the snapshots folder; otherwise, use your own trained model that is placed in the snapshots folder.

- Run the commands below to calculate Increase in Confidence (IC) and Average Drop (AD), if using the VGG-16 backbone:
~~~
cd scripts
sh VGG16_AD_IC.sh 
~~~

**OR**, if using the ResNet-50 backbone:
~~~
cd scripts
sh ResNet50_AD_IC.sh
~~~
Before running any of the .sh files, again set the img_dir, snapshot_dir, arch and percentage parameters inside the .sh file.

## Evaluation of the other methods
- To evaluate the methods that are used for comparison with L-CAM-Fm and L-CAM-Img, run the commands below to calculate Increase in Confidence (IC) and Average Drop (AD):
~~~
cd scripts
sh Inference_OtherMethods.sh 
~~~
Before running  the .sh file, first take the code for Grad-Cam, Grad-Cam++, Score-CAM and RISE from [ScoreCAM](https://github.com/yiskw713/ScoreCAM/blob/master/cam.py) repository and [RISE](https://github.com/eclique/RISE) repository  and save it to */L-CAM/utils/cam.py file. Than select from */L-CAM/Inference_OtherMethod the file with the method that you want to evaluate e.g. For ResNet-50 backbone and RISE method select ResNet50_rise.py from */L-CAM/Inference_OtherMethods folder/ and set it in the Inference_OtherMethods.sh file. Also, set the img_dir and percentage parameters inside the .sh file.
For example:
~~~
CUDA_VISIBLE_DEVICES=0 python ResNet_rise.py \
--img_dir='/ssd/imagenet-1k/ILSVRC2012_img_val' \
--percentage=0.5 \
~~~

## Parameters
During the training and evaluation stages the above parameters can be specified.

Parameter name | Description | Type |Default Value
| ---: | :--- | :---: | :---:
`--root_dir` | Root directory for the project. | str| ROOT_DIR |
`--img_dir` | Directory where the training images reside. | str| img_dir |
`--train_list` | The path where the annotations for training reside. | str| train_list |
`--test_list` | The path where the annotations for evaluation reside. | str| test_list |
`--batch_size` | Selected batch size. | int| Batch_size |
`--input_size` | Image scaling parameter: the small side of each image is resized, to become equal to this many pixels. | int| 256 |
`--crop_size` | Image cropping parameter: each (scaled) image is cropped to a square crop_size X crop_size pixels image. | int| 224 |
`--arch` | Architecture selected from the architectures that are avaiable in the models folder. | str | e.g. ResNet50_aux_ResNet18_TEST |
`--lr` | The initial learning rate. | float| LR |
`--epoch` | Number of epochs used in training process. | int| EPOCH |
`--snapshot_dir` | Directory where the trained models are stored. | str| Snapshot_dir |
`--percentage` | Percentage of saliency's muted pixels. | float| percent |

The above parameters can be changed in the .sh files. For example:
~~~
CUDA_VISIBLE_DEVICES=0 python Evaluation_L_CAM_ResNet50.py \
	--img_dir='/m2/ILSVRC2012_img_val' \
	--snapshot_dir='/m2/gkartzoni/L-CAM/snapshots/ResNet50_L_CAM_Img' \
	--arch='ResNet50_L_CAM_Img' \
	--percentage=0 \
~~~
We use relative paths for train_list and test_list so they are specified relative to the project path (/L-CAM) in the .py files. The paths that must be specified externally are arch(from */L-CAM/models folder), snapshot_dir, img_dir and percentage, as in the example.

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

I. Gkartzonika, N. Gkalelis, V. Mezaris, "Learning visual explanations for DCNN-based image classifiers using an attention mechanism", 2022, under review.
</div>

BibTeX:

```
@INPROCEEDINGS{9666088,
    author    = {Gkartzonika, Ioanna and Gkalelis, Nikolaos and Mezaris, Vasileios},
    title     = {Learning visual explanations for DCNN-based image classifiers using an attention mechanism},
    booktitle = {under review},
    month     = {},
    year      = {2022},
    pages     = {}
}
```

## License
<div align="justify">
    
Copyright (c) 2022, Ioanna Gkartzonika, Nikolaos Gkalelis, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
The training process is based on code released in the [DANet](https://github.com/xuehaolan/DANet) repository.

The code for the methods that are used for comparison with L-CAM-Fm and L-CAM-Img is taken from the [ScoreCAM](https://github.com/yiskw713/ScoreCAM/blob/master/cam.py) repository, except for the code for the RISE method, which is taken from the [RISE](https://github.com/eclique/RISE) repository.

<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreements H2020-951911 AI4Media and H2020-832921 MIRROR. </div>

