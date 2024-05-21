from glob import glob 
import pandas as pd
import sys
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from my_module.fileLocation import FileLocation
#from torchvision.datasets.folder import find_classes
#from torch import nn
#from my_module.model import model

# FileLocation 클래스의 인스턴스 생성
# https://github.com/Lornatang/GoogLeNet-PyTorch/blob/main/dataset.py


# 폴더 이름 정의
IMG_EXE=("jpg","jpeg","png","bmp")


os_name = sys.platform
fl = FileLocation()
fl.foldertoList_extraction('fire').chain_fileExten('jpg')
#result =fl.foldertoList_extraction('fire')

#& C:\anaconda3\envs\vsip\python.exe c:/AI_gitRep/frownFace/injun/g_main.py    


if os_name in 'win32':
    paths ="\\"
else:
    paths ="/"


print(len(fl[0]))
print(len(fl[1]))
print(len(fl[2]))
