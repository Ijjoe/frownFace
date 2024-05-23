import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
 
class inception_module(nn.Module):
    def __init__(self
                 ,in_channels: int
                 ,filters_1x1: int
                 ,filters_3x3_reduce:int
                 ,filters_3x3:int
                 ,filters_5x5_reduce:int
                 ,filters_5x5: int
                 ,filters_pool_proj:int):
        super().__init__()

        self.layout1 =nn.Sequential(
            nn.Conv2d(in_channels, filters_1x1,kernel_size=1,padding="same")
            ,nn.ReLU()
        )

        self.layout2= nn.Sequential(
            nn.Conv2d(in_channels, filters_3x3_reduce, kernel_size=1)
            ,nn.ReLU()
            ,nn.Conv2d(filters_3x3_reduce,filters_3x3, kernel_size=3, padding="same")
            ,nn.ReLU()
        )

        self.layout3 = nn.Sequential(
            nn.Conv2d(in_channels, filters_5x5_reduce, kernel_size=1)
            ,nn.ReLU()
            ,nn.Conv2d(filters_5x5_reduce,filters_5x5, kernel_size=5, padding="same")
            ,nn.ReLU()
        )

        self.layout4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
            ,nn.Conv2d(in_channels,filters_pool_proj, kernel_size=1, padding="same")
            ,nn.ReLU()
        )

    def _forward(self,x):
        layout1 = self.layout1(x)
        layout2 = self.layout2(x)
        layout3 = self.layout3(x)
        layout4 = self.layout4(x)

        outputs = [layout1,layout2,layout3,layout4]
        return outputs

    def forward(self,x):
        outputs = self._forward(x)
        return torch.cat(outputs,1)