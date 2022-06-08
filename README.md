# lung_covid
Colab and CNN Segmentation
# Lung and Covid-19 Segmentation With Deep Learning Models 
### Mehmet Furkan BAGCI A59007022
### And Kaan Ata YILMAZ A59009346


#Outline 
- Dataset 
- U-Net 
- U-Net Transfer Learning
- Deeplab v3
- Deeplab v3 Transfer learning
- Results 


For this project we used this dataset : COVID-QU-Ex[1]
Dataset contains 33,920 chest X-ray (CXR) images including their masks. The images and the masks are 256x256 size. Masks are contains 0's and 1's the images are gray images. 
There are 2 sections Infection Segmentation and Lung Segmentation For both section there are lung masks of the images for Infections Segmentation there are also infections masks that show only covi-19 infected locations. the data set includes as folder named Non-Covid which means pneomonia images, their covid infections masks are all black. Thanks to that the models will be able to distinguish pnueomonia from covid. 
For each section there are test, validation and train folders. While we are reaching those images and masks we used folders acording to their purposes.   
The structure of the dataset is: 
```
archive:
+---Infection Segmentation Data
|   \---Infection Segmentation Data
|       +---Test
|       |   +---COVID-19
|       |   |   +---images
|       |   |   +---infection masks
|       |   |   \---lung masks
|       |   +---Non-COVID
|       |   |   +---images
|       |   |   +---infection masks
|       |   |   \---lung masks
|       |   \---Normal
|       |       +---images
|       |       +---infection masks
|       |       \---lung masks
|       +---Train
|       |   +---COVID-19
|       |   |   +---images
|       |   |   +---infection masks
|       |   |   \---lung masks
|       |   +---Non-COVID
|       |   |   +---images
|       |   |   +---infection masks
|       |   |   \---lung masks
|       |   \---Normal
|       |       +---images
|       |       +---infection masks
|       |       \---lung masks
|       \---Val
|           +---COVID-19
|           |   +---images
|           |   +---infection masks
|           |   \---lung masks
|           +---Non-COVID
|           |   +---images
|           |   +---infection masks
|           |   \---lung masks
|           \---Normal
|               +---images
|               +---infection masks
|               \---lung masks
\---Lung Segmentation Data
    \---Lung Segmentation Data
        +---Test
        |   +---COVID-19
        |   |   +---images
        |   |   \---lung masks
        |   +---Non-COVID
        |   |   +---images
        |   |   \---lung masks
        |   \---Normal
        |       +---images
        |       \---lung masks
        +---Train
        |   +---COVID-19
        |   |   +---images
        |   |   \---lung masks
        |   +---Non-COVID
        |   |   +---images
        |   |   \---lung masks
        |   \---Normal
        |       +---images
        |       +---lung masks
        |       \---PennFudanPed
        |           +---PedMasks
        |           \---PNGImages
        \---Val
            +---COVID-19
            |   +---images
            |   \---lung masks
            +---Non-COVID
            |   +---images
            |   \---lung masks
            \---Normal
                +---images
                \---lung masks
```




33,920 chest X-ray (CXR) images
including their masks.
- The dataset consists of two parts:
- Lung Segmentation
- Covid-19 Infection Segmentation
- 2300 images have been used from
each section at the training (80% for
train 20% for validation)
- In the test section available inputs
(Covid-19: 1166; Lung:6788 ) have
been used


[1] Anas M. Tahir. Covid-qu-ex dataset. https://www.kaggle.com/datasets/
anasmohammedtahir/covidqu, Feb 2022.
