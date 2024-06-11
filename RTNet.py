import math
import torch
import torch.nn as nn
from PIL import Image


#卷积模块
class Conv(nn.Module):
    def __init__(self,in_c,out_c,k,s,p):
        super(Conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,k,s,p),
            # nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

#残差模块
class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.conv = nn.Sequential(
            Conv(64,128,3,1,1),
            Conv(128,256,3,1,1),
            Conv(256,256,3,1,1),
            Conv(256,128,3,1,1),
            Conv(128,64,3,1,1),)

    def forward(self, x):
        y = self.conv(x)
        return x+y


# 模型
class model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.illumination_net = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            ResidualBlock(),
            nn.Conv2d(64,1,3,1,1),
        )
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            ResidualBlock(),
            nn.Conv2d(64,1,3,1,1),
        )
    def forward(self,x):
        I = torch.sigmoid(self.illumination_net(x.to(torch.float32)))
        R = torch.sigmoid(self.reflectance_net(x.to(torch.float32)))
        return I,R

