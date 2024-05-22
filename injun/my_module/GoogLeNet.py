import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from my_module.inception_module import inception_module

class googlenet(nn.Module):
    def __init__(self
                 ,num_classes: int = 1000
                 ,aux_logits: bool = False
                 ,init_weights: Optional[bool] = None
                 ,dropout: float = 0.5
                 ,dropout_aux: float = 0.7):
        super().__init__()

        inception_block = inception_module

        self.aux_logits = aux_logits

        self.conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3,stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 =  nn.MaxPool2d(3,stride=2, ceil_mode=True)

        self.inception_3a = inception_block(in_channels=192
                                            ,filters_1x1=64
                                            ,filters_3x3_reduce=96
                                            ,filters_3x3=128
                                            ,filters_5x5_reduce=16
                                            ,filters_5x5=32
                                            ,filters_pool_proj=32
                                            )
        
        self.inception_3b = inception_block(in_channels=256
                                            ,filters_1x1=128
                                            ,filters_3x3_reduce=128
                                            ,filters_3x3=192
                                            ,filters_5x5_reduce=32
                                            ,filters_5x5=96
                                            ,filters_pool_proj=64
                                            )
        
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception_4a = inception_block(in_channels=480
                                            ,filters_1x1=192
                                            ,filters_3x3_reduce=96
                                            ,filters_3x3=208
                                            ,filters_5x5_reduce=16
                                            ,filters_5x5=48
                                            ,filters_pool_proj=64
                                            )
        if aux_logits:
            self.auxilliary_output_1 = nn.Sequential(nn.AvgPool2d(5, stride=3)
                                                     ,nn.Conv2d(512,128, kernel_size=1,padding="same")
                                                     ,nn.ReLU()
                                                     ,nn.Flatten()
                                                     ,nn.Linear(2048,1024)
                                                     ,nn.ReLU()
                                                     ,nn.Dropout(p=0.7)
                                                     ,nn.Linear(1024, num_classes)
                                                     )
        else:
            self.auxilliary_output_1 = None




        self.inception_4b = inception_block(in_channels=512
                                            ,filters_1x1=160
                                            ,filters_3x3_reduce=112
                                            ,filters_3x3=224
                                            ,filters_5x5_reduce=24
                                            ,filters_5x5=64
                                            ,filters_pool_proj=64
                                            )
        
        self.inception_4c = inception_block(in_channels=512
                                            ,filters_1x1=128
                                            ,filters_3x3_reduce=128
                                            ,filters_3x3=256
                                            ,filters_5x5_reduce=24
                                            ,filters_5x5=64
                                            ,filters_pool_proj=64
                                            )
        
        self.inception_4d = inception_block(in_channels=512
                                            ,filters_1x1=112
                                            ,filters_3x3_reduce=114
                                            ,filters_3x3=288
                                            ,filters_5x5_reduce=32
                                            ,filters_5x5=64
                                            ,filters_pool_proj=64
                                            )
        
        if aux_logits:
            self.auxilliary_output_2 = nn.Sequential(nn.AvgPool2d(5, stride=3)
                                                     ,nn.Conv2d(528,128, kernel_size=1,padding="same")
                                                     ,nn.ReLU()
                                                     ,nn.Flatten()
                                                     ,nn.Linear(2048,1024)
                                                     ,nn.ReLU()
                                                     ,nn.Dropout(p=0.7)
                                                     ,nn.Linear(1024, num_classes)
                                                     )
        else:
            self.auxilliary_output_2 = None



        self.inception_4e = inception_block(in_channels=528
                                            ,filters_1x1=256
                                            ,filters_3x3_reduce=160
                                            ,filters_3x3=320
                                            ,filters_5x5_reduce=32
                                            ,filters_5x5=128
                                            ,filters_pool_proj=128
                                            )
        
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
                                                     
        self.inception_5a = inception_block(in_channels=832,
                                             filters_1x1=256,
                                             filters_3x3_reduce=160,
                                             filters_3x3=320,
                                             filters_5x5_reduce=32,
                                             filters_5x5=128,
                                             filters_pool_proj=128,
                                            )
        self.inception_5b = inception_block(in_channels=832,
                                             filters_1x1=384,
                                             filters_3x3_reduce=192,
                                             filters_3x3=384,
                                             filters_5x5_reduce=48,
                                             filters_5x5=128,
                                             filters_pool_proj=128,
                                            )
        
        self.avgpool = nn.AvgPool2d((7,7), stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor):
        x =self.conv1(x)
        x = self.maxpool1(x)
        x =self.conv2(x)
        x =self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool3(x)
        x = self.inception_4a(x)

        aux1: Optional[Tensor] = None
        if self.auxilliary_output_1 is not None:
            aux1 = self.auxilliary_output_1(x)

        x = self.inception_4b(x)
        # N x 512 x 14 x 14
        x = self.inception_4c(x)
        # N x 512 x 14 x 14
        x = self.inception_4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.auxilliary_output_2 is not None:
            aux2 = self.auxilliary_output_2(x)

        x = self.inception_4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception_5a(x)
        # N x 832 x 7 x 7
        x = self.inception_5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux1, aux2
