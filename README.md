# Deblur-GAN-LP-Recognition
using deblur gan for improving LP Recognition in YOLO-V3

## DISCRIPTION
> in this project we use deblur gan structure for imporving vehicle images lead to improve license-plate boxes in images,after that we used YOLO-V3 and for LP recognition that was used before in my github repo.so as preprocess step we used deblur gan for image quality enhancement,we found that the accuracy of LP rcognition and Character recognition will increase by using this stage,because the blur that was build by camera moving or hand shaking or another things lead to degrade the LP and Char in vehicle images,so by deblur processing we sharp our dataset iamges and will get better performance in LP-Recognition task.

## Dataset and Preprocess
> for preparing dataset,i have walked alon road,highway in fereydounkenar city.by using my camera phone i have captured 78 videos and by using software i stored about ~4000 frames among with them,further more for training procedure we used 2 images with same name because we need the iamge and its blur format,So as a preprocess stage we designed a code for creating blur images related to our real dataset.with below code you will enable to create blure images,with different blur kerne size as `3*3`,`5*5`,`9*9` and by knowing that the image inpute size for input stage we resize them in size `1280*270`,in this structure you can use any kernel blur that you want,set the `inputfolder_path` and `outputfolder_path` basis on your `path` in your environment:
```
import numpy as np
import cv2
import glob
import os
from os import path 
from PIL import Image

# You can change the kernel size as you want 
#blurImg = cv2.blur(img,(5,5)) 
#blurImg = cv2.GaussianBlur(img, (9,9),0)  
#blurImg = cv2.medianBlur(img,9)  
#blurImg = cv2.bilateralFilter(img,9,75,75)

inputfolder_path="/content/drive/MyDrive/validation"
outputfolder_path="/content/drive/MyDrive/validation-7*7"

i=1
for img in glob.glob(inputfolder_path+"/*.jpg"):
  name=path.splitext(os.path.basename(img))[0]
  image=cv2.imread(img)
  image=cv2.resize(image,(1280,720))
  blurImg = cv2.blur(image,(7,7))
  cv2.imwrite(outputfolder_path+'/'+name+'.jpg',blurImg)
  print(i)
  i=i+1
```

> after constructing your blur iamges,you must split these images in three groups as `train` , `test` and `valid` so,i splitted them as ~3600 images for train,400 iamges for test and 100 images for vlidation,in three directories:
```
*  dataset/train/images/blur && dataset/train/images/blur
*  dataset/test/images/blur && dataset/test/images/blur
*  dataset/valid/images/blur && dataset/valid/images/blur
```
> HINT:due to large size of dataset,i have not able to set them in github,so if you need my dataset,please contact me via my email address.

## MODEL
> we have 4 types for applying our model :
* Single-Scale Network without long-skip connection
* Single-Scale Network with long-skip connection
* Multi-Scale Network without long-skip connection
* Multi-Scale Network with long-skip connection
> you see the 4 conditions of this architecture,in below image:
![architecture](https://user-images.githubusercontent.com/53394692/111139212-180c9b00-8596-11eb-8782-a11dd1647655.jpg)
> after training over these models,i found that the (Multi-Scale Network with long-skip connection) had better PSNR,SSIM,MSSIM so i selected this model as deblur module.

## Training Procedures
* 1.mount your google drive and change your `dir` basis on position you set the `Deblur-GAN-LP-Recognition` directory :
```
from google.colab import drive
drive.mount('/content/drive/')
import os
os.chdir("/content/drive/MyDrive/Pytorch-Image-Deblurring-master-run7")
!ls
```
* 2.install your requirements in google colab baisis on `requirements.txt` file in directory:
```
!pip install -r requirements.txt
```
* 3.training and testing different conditions :
> * for train/test multiskip :
```
#!python main.py --gpu 0 --multi --skip --exp_name multi_skip --batch_size 16 --epochs 400 --finetuning
#for testing
#!python test.py --gpu 0 --exp_name multi_skip --padding 0
#for applying model
#!python demo.py --gpu 0 --train_dir pretrained --exp_name multi_skip --image dataset/test/images/blur/*.jpg
```
> * for train/test multi-noskip :
```
#!python main.py --gpu 0 --multi --exp_name multi_noskip --batch_size 16 --epochs 400 --finetuning
#for testing
#!python test.py --gpu 0 --exp_name multi_noskip --padding 0
#for applying model
#!python demo.py --gpu 0 --train_dir pretrained --exp_name multi_noskip --image dataset/test/images/blur/*.jpg
```
> * for train/test single-skip :
 ```
#!python main.py --gpu 0 --skip --exp_name single_skip --batch_size 16 --epochs 400 --finetuning
#for testing
#!python test.py --gpu 0 --exp_name single_skip --padding 0
#for applying model
#!python demo.py --gpu 0 --train_dir pretrained --exp_name single_skip --image dataset/test/images/blur/*.jpg
 ```
> * for train/test single-noskip :
```
#!python main.py --gpu 0 --exp_name single_noskip --batch_size 16 --epochs 400 --finetuning
#for testing
#!python test.py --gpu 0 --exp_name single_noskip --padding 0
#for applying model
#!python demo.py --gpu 0 --train_dir pretrained --exp_name single_noskip --image dataset/test/images/blur/*.jpg
```
> for example,













