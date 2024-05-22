from torch.utils.data import Dataset
from torchvision import transforms
import module.DataPreprocessing as data_pp


# 커스텀 데이터셋 만들기
class CustomDataset(Dataset):
    
    # param
        # 데이터셋 파일경로 (list)
        # 인덱싱된 라벨 (list)
        # 조정 사이즈 (int)
    def __init__(self, file_path, indexed_label, img_size):
        
        self.file_path = file_path
        self.indexed_label = indexed_label
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dp = data_pp.DataPreprocessing()
        self.img_size = img_size

    
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
        
        # 사이즈 조정 및 패딩
        img = self.dp.resize_image(img_path, self.img_size)
        img = self.dp.padding_image(img, self.img_size)
        
        # 텐서로 변환
        img = self.transform(img)

        return img, label