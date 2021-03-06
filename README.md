# Lung and Covid-19 Segmentation With Deep Learning Models 
## The First Part U-Net, DeepLab V3+
### Mehmet Furkan BAGCI A59007022

# How to Run the code for sections 
Our github repo is created with Colab and any method but colab takes more time. Code calls the dataset from Kaggle , functions from this Github repo, models from Google Drive. There are 8 train files, 1 test file, 1 plot file. Possible problem about the code is the memory of the GPU
## Outline 
- Train the models 
- Test the models 
- Plot the training process 

### Train models 
For this project we used this dataset : COVID-QU-Ex[1] Dataset contains 33,920 chest X-ray (CXR) images including their masks. 
To call and unzipping the dataset takes 3 mins. 
There are atleast 8 .ipynb file for training, 3 feature model(U-Net, DeepLab v3+), Class(Lung, Covid), method(transfer learning, pretrained) 2*2*2=8  

This functions are in side of the files.
Call this repo for functions 
```
! git clone https://github.com/mbagci06/lung_covid
```
Call library of Kaggle
```
! pip install kaggle
```
Reach the kay file of mine for kaggle and call our dataset
```
! mkdir ~/.kaggle
! cp lung_covid/kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d anasmohammedtahir/covidqu
```
unzip the dataset to archive folder
```
!unzip "/content/covidqu.zip" -d "/content/archive/" 
```
The functions and modules for training.
```
%run lung_covid/functions.ipynb
%run lung_covid/DeepLab.ipynb

```
There might be problem about GPU memory if it is lees then 16GB to fix it change this line and make batch_size smaller.
```
batch_size=25
```

The at the end of the code there is saving modules for model and loss,accuracy values. to save them downlaod to your computer or send to your Drive(faster for models) 

### Test the models 
The same way we called the Dataset we need to call it again addition to that we have models this time. 
```
gdown helps to download Google Drive files 
```
!pip install gdown

This folder has the trained models some of them about 670Mb, downloading from Drive to colab is the fastest. 
```
!gdown --folder https://drive.google.com/drive/folders/1MIzhxSou4TRtQTcG3AADqLag28nj3FrV?usp=sharing
```
## Plot the training process
As we called the models we can call the txt files which are created in the training section. 
To plot we need to call the functions that used in the training. 

# The files and their duties 
- functions.ipynb : Accuracy, dataloader, dataset, pre-process  functions
- unet.ipynb : function and models related to U-Net 
- DeepLab.ipynb: function and models related to DeepLab v3+
- test_results.ipynb : accuracy results for models.
### Train files : 
DeepLab_train_covid.ipynb 
DeepLab_train_lung.ipynb 
DeepLab_transfer_covid.ipynb 
DeepLab_transfer_train_lung.ipynb 
Unet_NONORM_covid.ipynb 
Unet_train_lung.ipynb 
Unet_transfer_train_covid.ipynb 
Unet_trasnfer_train_lung.ipynb
Unet_train_covid.ipynb

# Second part of the project The Mask R-CNN part of the repo 
## And Kaan Ata YILMAZ A59009346
COdes best runs in the colab
To run the Mask R-Cnn files:
First Kaggle dataset should be downloaded https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
and should be named archive. The code will reach it.

```
 def __init__(self,root_dir = ("/content/archive"),class_folder='lung',split = "Train", transforms = None , shuffle = True,batch_size=1,
                 sample_size=500 ):
```


In the MaskR-CNN file
This line calls the function.ipynb but it is in the same folder

```
!cp /content/drive/MyDrive/Covid/functions.ipynb  /content  
```

In the class kaan_dataset() there is a root_dir= value 

```
 def __init__(self,root_dir = ("/content/archive"),class_folder='lung',split = "Train", transforms = None , shuffle = True,batch_size=1,
                 sample_size=500 ):
```

## References
- [1] Pytorch. (n.d.). Pytorch/vision: Datasets, transforms and models specific to Computer Vision. Retrieved June 8, 2022, from https://github.com/pytorch/vision
- [2] VainF. ???VAINF/DeepLabV3Plus-Pytorch: Deeplabv3 and deeplabv3+ with Pretrained Weights for Pascal Voc &amp; Cityscapes.??? GitHub. Accessed June 8, 2022. https://github.com/VainF/DeepLabV3Plus-Pytorch. 
- [3] Milesial. (n.d.). Milesial/Pytorch-UNet: Pytorch implementation of the U-Net for image semantic segmentation with high quality images. Retrieved June 8, 2022, from https://github.com/milesial/Pytorch-UNet
- [4] Torchvision object detection finetuning tutorial??. TorchVision Object Detection Finetuning Tutorial - PyTorch Tutorials 1.11.0+cu102 documentation. (n.d.). Retrieved June 9, 2022, from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
