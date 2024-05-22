from PIL import Image
import numpy as np
import torch

# 서비스 구현 클래스
class Service:
    
    def __init__(self):
        pass
    
    # 전체 위성사진 훈련 이미지 사이즈로 분할 및 모델 예측
    # param
        # 모델 (model)
        # 테스트 위성사진 경로 (str)
        # 학습시 사용한 이미지 사이즈 (int)
    def eval_in_realmap(self, model, img_path, img_size):
        pic = np.array(Image.open(img_path))
        pic = np.moveaxis(pic, 2, 0)
        
        result_lst = []
        
        for y in range(pic.shape[1] - img_size + 1): # 세로축 (행)
            for x in range(pic.shape[2] - img_size + 1): # 가로축 (열)
                img = pic[:3, y:y+img_size, x:x+img_size]
                img = torch.Tensor(img.reshape(1, 3, img_size, img_size))
                result = model(img)
                result_lst.append(torch.max(result, 1)[1].item())             
        
        return result_lst
    
    