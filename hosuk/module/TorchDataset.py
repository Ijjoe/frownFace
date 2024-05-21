from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
from torchvision.transforms.functional import pad
import matplotlib.pyplot as plt
import random

# 라이브러리 사용해서 커스텀 데이터셋 만들기
class TorchDataset(Dataset):
    
    # param
        # 데이터셋 루트 폴더 경로 (str)
        # 찾을 확장자명
        # 조정 사이즈 (int)
    def __init__(self, root_folder, extension, img_size):
        
        self.img_size = img_size
        
        # 루트 폴더에서 
        self.root_folder = root_folder
        self.file_path, self.target = self.get_file_path(extension)
        self.indexed_label = self.label_indexing(self.target) # 라벨 인덱싱
        
        # 이미지 사이즈 변환
        # resize의 경우 인자를 하나만 넣어야 비율에 맞춰서 변환해줌
        self.transform = transforms.Compose([transforms.ToTensor()])
        

    # 루트폴더에서 파일 리스트(X), 라벨(Target) 가져오기
    # param
        # 찾을 파일 확장자명 (str)
    # return : 전체 파일경로(list), 폴더명(라벨, list)
    def get_file_path(self, extension):

        #
        self.root_folder = self.root_folder + "**\\**." + extension
        file_list = glob(self.root_folder, recursive=True)

        file_path = []
        target = []

        for path in file_list:
            
            split_path = path.split("\\")
            folder_name = split_path[-2]
            
            if folder_name.split()[-1] != "GT":
                file_path.append(path)

                # 오타로 인한 다른 폴더 이름 동일하게 처리
                if folder_name == "Gilt Head Bream":
                    folder_name = "Gilt-Head Bream"
                elif folder_name == "Horse Mackerel":
                    folder_name = "Hourse Mackerel"
                
                target.append(folder_name)
                
        return file_path, target  
        
    
    # 타겟 라벨 인덱싱
    # param
        # 타겟 리스트 (list)    
    # return : 인덱싱 된 라벨 리스트 (list)
    def label_indexing(self, target_list):

        # 인덱스 딕셔너리 만들기        
        s = set(target_list)
        self.label_dict = {label : i for i, label in enumerate(sorted(list(s)))}
        
        # 인덱싱된 라벨 리스트 만들기
        indexed_label = [self.label_dict[target] for target in target_list]
        
        return indexed_label

    
    # 라벨 - 인덱스 딕셔너리 보기
    # return : 딕셔너리 (dict)
    def get_label_dict(self):
        return self.label_dict
    
    
    # 이미지 리사이즈
    # param
        # 이미지 파일 경로 (str)
        # 최대 이미지 사이즈 (int)
    # return : 리사이즈된 PIL open 이미지
    def resize_image(self, image_file_path):
    
        image = Image.open(image_file_path)
        
        # 원본 이미지의 가로(행), 세로(열) 길이
        origin_h, origin_w = image.size[0], image.size[1]
        
        # 가로세로 중 더 큰게 기준이 됨 (애초에 작으면 그냥 그대로 리턴)
        standard = origin_h if origin_h >= origin_w else origin_w
        if standard <= self.img_size:
            return image
        
        # 비율을 정함
        ratio = self.img_size / standard
        new_h = int(origin_h * ratio)
        new_w = int(origin_w * ratio)
        
        return image.resize((new_h, new_w))
    
    
    # 이미지 패딩
    # param
        # 이미지 파일 경로(str)
        # 최대 이미지 사이즈 (int)
    # return : 패딩된 이미지
    def padding_image(self, image):
        
        # 이미지 사이즈
        h, w = image.size[0], image.size[1]

        # 패딩 사이즈
        pad_h = int((self.img_size - h) / 2)
        h_rest = int((self.img_size - h) % 2) # 나머지
        pad_w = int((self.img_size - w) / 2)
        w_rest = int((self.img_size - w) % 2)
        
        # 사이즈 홀수라 나머지 있을 경우 아랫쪽, 오른쪽에 더해줌
        pad_size = (pad_h, pad_w, pad_h+h_rest, pad_w+w_rest)
        
        # 패딩
        return pad(image, pad_size, fill = 0)


    # 샘플 이미지 출력하기
    # param
        # 출력 이미지 갯수(int)
        # 랜덤 여부 (bool = True)
    # return : 그림 출력
    def get_sample_image(self, sample_count, isRandom = True):
        
        # 행렬 크기
        figure = plt.figure(figsize = (12,6))
        row = sample_count // 4 if sample_count % 4 == 0 else sample_count // 4 + 1
        col = 4
        
        for i in range(1, sample_count + 1):
            
            # 랜덤이 아닐 경우는 인덱스 0부터 갯수대로 출력
            r_idx = random.randint(0, len(self.file_path)) if isRandom else i - 1
            img = Image.open(self.file_path[r_idx])
            label = self.target[r_idx]
            figure.add_subplot(row, col, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img, cmap = "jet")
        
        plt.show()
    
    
    def get_target_ratio(self):
        pass
    
    
    # 전체 데이터셋 길이
    # return : 길이 (int)
    def __len__(self):
        return len(self.file_path)
    
    
    # 인덱스에 따라 이미지 뽑아서 텐서로 변환
    # param
        # 인덱스
    # return : 텐서변환된 이미지 (tensor), 라벨(list)
    def __getitem__(self, idx):
        
        img_path = self.file_path[idx]
        label = self.indexed_label[idx]
        
        img = self.resize_image(img_path) # 리사이즈
        img = self.padding_image(img) # 패딩
        img = self.transform(img) # 스케일링 및 텐서변환
        
        return img, label