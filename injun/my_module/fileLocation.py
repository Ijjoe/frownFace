import numpy as np 
from glob import glob 
import os


# FileLocation.py
class FileLocation:
    def __init__(self):
        self.result = 0
 

    def foldertoList_extraction(self,dv):
        try:
            ret =[]
            folderlist=[]
            listcount=[]
            file_path = glob(f'{os.getcwd()}\\{dv}\\**\\', recursive=True)
            for file in file_path:
                listcount.append(len(file.split('\\')))
                folderlist.append(file)
                
            b = np.array(listcount)   #리스트에 특정 중복값 넘파이 변환하여 
            nplist=np.where(b == max(listcount))  #지정된 인덱스만 전체 추출
            indices = nplist[0].tolist()

            for i in indices:
                ret.append(folderlist[i])
        except FileNotFoundError as e:
            print("지정된 디렉토리가 존재하지 않습니다:", e)
        except:
            print('경로 확인 요망 / 파일 위치 확인 / 예외가 발생했습니다.')
            
        return self.ret
