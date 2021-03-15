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

> after constructing your blur iamges,you





