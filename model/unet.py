import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model.unet_utils import *
class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out
    

class UNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        
        self.enc1 = Block(in_channels, 64, 64, batch_norm)
        self.enc2 = Block(64, 128, 128, batch_norm)
        self.enc3 = Block(128, 256, 256, batch_norm)
        self.enc4 = Block(256, 512, 512, batch_norm)
        
        self.center = Block(512, 1024, 512, batch_norm)
        
        self.dec4 = Block(1024, 512, 256, batch_norm)
        self.dec3 = Block(512, 256, 128, batch_norm)
        self.dec2 = Block(256, 128, 64, batch_norm)
        self.dec1 = Block(128, 64, 64, batch_norm)
        
        self.out = torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down(enc1))
        enc3 = self.enc3(self.down(enc2))
        enc4 = self.enc4(self.down(enc3))
        
        center = self.center(self.down(enc4))
        
        dec4 = self.dec4(torch.cat([self.up(center, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out
    

class PretrainedUNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        
        self.init_conv = torch.nn.Conv2d(in_channels, 3, 1)
        
        endcoder = torchvision.models.vgg11(pretrained=True).features
        self.conv1 = endcoder[0]   # 64
        self.conv2 = endcoder[3]   # 128
        self.conv3 = endcoder[6]   # 256
        self.conv3s = endcoder[8]  # 256
        self.conv4 = endcoder[11]   # 512
        self.conv4s = endcoder[13]  # 512
        self.conv5 = endcoder[16]  # 512
        self.conv5s = endcoder[18] # 512
    
        self.center = Block(512, 512, 256, batch_norm)
        
        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)
        
        self.out = torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x):  
        init_conv = torch.nn.functional.relu(self.init_conv(x), inplace=True)

        enc1 = torch.nn.functional.relu(self.conv1(init_conv), inplace=True)
        enc2 = torch.nn.functional.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3s(enc3), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4s(enc4), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5s(enc5), inplace=True)
        
        center = self.center(self.down(enc5))
        
        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], 1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out
# class UNet(nn.Module):
#     """
#     U-Net: Convolutional Networks for Biomedical Image Segmentation
#     (Ronneberger et al., 2015)
#     https://arxiv.org/abs/1505.04597

#     Contracting Path
#         - Two 3x3 Conv2D (Unpadded Conv, i.e. no padding)
#         - followed by a ReLU
#         - A 2x2 MaxPooling (with stride 2)
#     Expansive Path : sequence of "up-convolutions" and "concatenation" with high-resolution feature from contracting path
#         - "2x2 up-convolution" that halves the number of feature channels
#         - A "concatenation" with the correspondingly cropped feature map from the contracting path
#         - Two 3x3 Conv2D
#         - Followed by a ReLU

#     Final Layer
#         - "1x1 Conv2D" is used to map each 64 component feature vector to
#         the desired number of classes
#     """
#     def __init__(self, n_channels, n_classes , bilinear = False):
#         super(UNet,self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.in_conv = UNetConvBlock(self.n_channels , 64)
#         self.Down1 = Down(64 , 128)
#         self.Down2 = Down(128, 256)
#         self.Down3 = Down(256, 512)
#         self.Down4 = Down(512, 512)
#         self.Up1 = Up(512 + 512, 256 , self.bilinear)
#         self.Up2 = Up(256 + 256, 128 , self.bilinear)
#         self.Up3 = Up(128 + 128 , 64 , self.bilinear)
#         self.Up4 = Up(64 + 64, 64 , self.bilinear)
#         self.out_conv = OutConv(64, n_classes)

#     def forward(self,x):
#         x1 = self.in_conv(x)
#         x2 = self.Down1(x1)
#         x3 = self.Down2(x2)
#         x4 = self.Down3(x3)
#         x5 = self.Down4(x4)
#         x = self.Up1(x5,x4)
#         x = self.Up2(x ,x3)
#         x = self.Up3(x ,x2)
#         x = self.Up4(x ,x1)
#         out = self.out_conv(x)
#         return out


if __name__ == '__main__':
    UNet(3,10)

