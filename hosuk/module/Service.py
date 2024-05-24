from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

# 서비스 구현 클래스
class Service:
    
    def __init__(self):
        pass
    
    # 전체 위성사진 훈련 이미지 사이즈로 분할 및 모델 예측
    # param
        # 모델 (model)
        # 테스트 위성사진 경로 (str)
        # 학습시 사용한 이미지 사이즈 (int)
    def eval_in_realmap(self, model, img_path, img_size, device):
        
        pic = np.array(Image.open(img_path))
        
        print("원본크기 :", pic.shape) # 테스트 이미지 사이즈 확인
        pic = np.moveaxis(pic, 2, 0) # 채널 순서 변경
                
        transform = transforms.Compose([transforms.ToTensor()])
        for y in tqdm(range(pic.shape[1] - img_size + 1)): # 세로축 (행)
            
            for x in range(pic.shape[2] - img_size + 1): # 가로축 (열)
                
                # 모델에 적용된 이미지 크기만큼 잘라줌
                img = pic[:3, y:y+img_size, x:x+img_size]
                
                # 다시 채널 변경 후 transforms 적용
                img = np.moveaxis(img, 0, 2)
                img = transform(img)
                
                # 이미지 shape 변경 (모델 적용을 위해 4차원으로)
                img = img.reshape(1, 3, img_size, img_size)
                img = img.to(device)
                
                # 모델 평가
                result = model(img)
                
                # 결과 (0 or 1)
                final_result = torch.max(result, 1)[1].item()
                
                # 결과가 1로 나올 경우
                if final_result == 1:
                    print("산불!!!!")
                    break
            else:
                continue
            print("산불안남!!")    
            break