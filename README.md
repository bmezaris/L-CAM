# L-CAM

This repository hosts the code and data lists for our two learning-based eXplainable AI (XAI) methods called L-CAM-Fm and L-CAM-Img, for deep convolutional neural networks (DCNN) image classifiers. Our methods receive as input an image and a class label and produce as output the image regions that the DCNN has looked at in order to infer this class. Both methods use an attention mechanism (AM),  trained  end-to-end  along  with the original (frozen) DCNN, to derive class activation maps (CAMs)from the last convolutional layer’s feature maps (FMs). During training, CAMs are applied to the FMs (L-CAM-Fm) or the input image(L-CAM-Img) forcing the AM to learn the image regions explaining the DCNN’s outcome. Two widely used evaluation metrics, Increase in Confidence (IC) and Average Drop (AD), are used for evaluation.
- This repository contains the code for training L-CAM-Fm and L-CAM-Img, using VGG16 or ResNet-50 as the pre-trained backbone network along with the Attention Mechanism and our selected loss function. There is also code to train the above networks with the conventional cross-entropy loss. The models are named as following:
First the name of the backbone(ResNet50 or VGG16) and after the method's name(L-CAM-Fm or L-CAM-Img). If the model uses the cross-entropy loss(not our selected loss function) there is also an A character  at the end of the model. e.g.  ResNet50_L_CAM_ImgA.py There is also the variation with VGG16 backbone and L-CAM-Img method with AM's input the 7×7 FMs(after the last max pooling layer of VGG-16), not the last convolutional layer as in all the others models.  
- Instead of training, the user can also download the pre-trained models for L-CAM-Fm and L-CAM-Img(again using VGG16 or ResNet-50 as the backbone network along with the Attention Mechanism and our selected loss function [here](https://drive.google.com/drive/folders/1QiwB3iEobEPnSB9NRSsmDaUAuBMiPdz2?usp=sharing). The pre-trained models are named as their model name(as explained in the previous paragraph). 
- There is also code for evaluating our method according to two widely used evaluation metrics for DCNN explainability, Increase in Confidence (IC) and Average Drop (AD)
- In [L-CAM/datalist/ILSVRC](https://github.com/gkartzoni/L-CAM/tree/main/datalist/ILSVRC) there are text files with annotations for training VGG16 and ResNet-50 (VGG16_train.txt, ResNet50_train.txt) and text files with annotations for 2000 randomly selected images to be used at the evaluation stage (VGG16_2000.txt, ResNet50_2000.txt) for the L-CAM method.
- The ImageNet1K dataset images should be downloaded by the user manually.
- This project is implemented and tested in python 3.6 and PyTorch 1.9.

## Visual examples (and comparison) of results 
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

- To train from scratch VGG16 or ResNet-50, run for the VGG16 backbone and the selected loss function:
~~~
cd scripts
sh VGG16_train.sh 
~~~
**OR**, for the VGG16 backbone and cross-entropy loss:
~~~
cd scripts
sh VGG16_train_CE.sh 
~~~
**OR**, for the ResNet-50 backbone:
~~~
cd scripts
sh ResNet50_train.sh
~~~
**OR**, for the ResNet-50 backbone and cross-entropy loss:
~~~
cd scripts
sh ResNet50_train_CE.sh 
~~~
Before running any of the .sh files, set the img_dir, snapshot_dir and arch parameters inside the .sh file. The produced model will be saved in the snapshots folder. 

## Evaluation
- To evaluate the model, download the pretrained models that are available in this [GoogleDrive](https://drive.google.com/drive/folders/1QiwB3iEobEPnSB9NRSsmDaUAuBMiPdz2?usp=sharing), and place the downloaded folders (VGG16_L_CAM_Img, VGG16_L_CAM_Fm, VGG16_7x7_L_CAM_Img, ResNet50_L_CAM_Fm, ResNet50_L_CAM_Img) in the snapshots folder; otherwise, use your own trained model that is placed in the snapshots folder.

- Run the commands below to calculate Increase in Confidence (IC) and Average Drop (AD), if using the VGG16 backbone:
~~~
cd scripts
sh VGG16_AD_IC.sh 
~~~

**OR**, if using the ResNet-50 backbone:
~~~
cd scripts
sh ResNet50_AD_IC.sh
~~~
Before running any of the .sh files, again set the img_dir, snapshot_dir and arch parameters inside the .sh file.

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

The above parameters can be changed in the .sh files. For example:
~~~
CUDA_VISIBLE_DEVICES=0 python Evaluation_L_CAM_VGG16.py \
--arch=VGG16_L_CAM_Fm \
--img_dir='/ssd/imagenet-1k/ILSVRC2012_img_val' \
~~~
We use relative for train_list and test_list so they are specified relatively to the project path (/L-CAM) in the .py files. The paths that must be specified externally is arch(from models' folder), snapshot_dir and img_dir, as in the example. If the images are saved in the dataset folder, set --img_dir=path2datasetFolder/ILSVRC2012_img_train for the training stage and --img_dir=path2datasetFolder/ILSVRC2012_img_val for the evaluation stage inside the .sh files. Same for snapshot_dir and arch paramenters.

# Acknowledgement
The training process is based on code released in the [DANet](https://github.com/xuehaolan/DANet) repository.

If you find our L-CAM code useful, please cite the following paper where our method is reported:

I. Gkartzonika, N. Gkalelis, V. Mezaris, "Learning visual explanations for DCNN-based image classifiers using an attention mechanism", under review.



