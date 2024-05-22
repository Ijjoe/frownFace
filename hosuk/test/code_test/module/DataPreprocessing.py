from glob import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.transforms.functional import pad
import torch
from random import shuffle
from tqdm import tqdm


# Data 전처리 클래스
class DataPreprocessing:
    
    def __init__(self):
        
        # 라벨 인덱싱 저장할 변수 지정
        self.__label_dic = {}


    # 라벨 인덱싱 딕셔너리 리턴
    def get_label_dic(self):
        return self.__label_dic


    # 지정 폴더 안 모든 확장자에 대한 파일 찾기
    # param
        # 루트 폴더(str)
        # 찾을 파일 확장자(str)
    # return : 파일 전체 루트(list), 라벨(list)
    def get_file_path(self, folder_path, extension):
   
        # 폴더 안에 하나라도 있는지 확인
        f_lst = glob(folder_path + "\\*")
        count = len(f_lst)
        
        # 폴더 안 파일경로 저장 리스트
        file_path, label = [], []
        
        # 내용이 하나라도 있거나 GT로 끝나는 폴더가 아닐때 
        if count > 0 and os.path.basename(folder_path).split()[-1] != "GT":

            # 안에 있는 폴더 or 파일 리스트
            for f in f_lst:
                
                # f가 폴더면 다시 재귀 수행
                if os.path.isdir(f):
                    
                    # 재귀 수행 뒤 받은 리스트를 file_path에 합쳐줌
                    lst1, lst2 = self.get_file_path(f, extension)
                    file_path.extend(lst1)
                    label.extend(lst2)

                # f가 파일이면서 확장자가 맞으면 리스트에 넣어줌
                elif os.path.basename(f).split(".")[-1] == extension:
                    file_path.append(f)
                    
                    # 라벨(폴더 이름) 추가
                    folder_name = f.split("\\")[-2]
                    
                    # 데이터 두 개 폴더가 오타로 인해 이름이 달라져서 교정                    
                    if folder_name == "Gilt Head Bream":
                        folder_name = "Gilt-Head Bream"
                    elif folder_name == "Horse Mackerel":
                        folder_name = "Hourse Mackerel"
                    
                    label.append(folder_name)
                
        # 최종 파일 리스트 리턴
        return file_path, label
    
    
    # 샘플 이미지 출력하기
    # param
        # 출력 이미지 갯수(int)
        # 파일 경로 리스트(list)
        # 라벨 리스트(list)
        # 랜덤 여부 (bool = True)
    # return : 그림 출력
    def get_sample_image(self, sample_count, file_path, label_lst, isRandom = True):
        
        # 행렬 크기
        figure = plt.figure(figsize = (12,6))
        row = sample_count // 4 if sample_count % 4 == 0 else sample_count // 4 + 1
        col = 4
        
        for i in range(1, sample_count + 1):
            
            # 랜덤이 아닐 경우는 인덱스 0부터 갯수대로 출력
            r_idx = random.randint(0, len(file_path)) if isRandom else i - 1
            img = Image.open(file_path[r_idx])
            label = label_lst[r_idx]
            figure.add_subplot(row, col, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img, cmap = "jet")
        
        plt.show()
            
        
    # 이미지 리사이즈
    # param
        # 이미지 파일 경로 (str)
        # 최대 이미지 사이즈 (int)
    # return : 리사이즈된 PIL open 이미지
    def resize_image(self, image_file_path, target_size):
    
        image = Image.open(image_file_path)
        
        # 원본 이미지의 가로(행), 세로(열) 길이
        origin_h, origin_w = image.size[0], image.size[1]
        
        # 가로세로 중 더 큰게 기준이 됨 (애초에 작으면 그냥 그대로 리턴)
        standard = origin_h if origin_h >= origin_w else origin_w
        if standard <= target_size:
            return image
        
        # 비율을 정함
        ratio = target_size / standard
        new_h = int(origin_h * ratio)
        new_w = int(origin_w * ratio)
        
        return image.resize((new_h, new_w))


    # 이미지 패딩
    # param
        # 이미지 파일 경로(str)
        # 최대 이미지 사이즈 (int)
    # return : 패딩된 이미지
    def padding_image(self, image, target_size):
        
        # 이미지 사이즈
        h, w = image.size[0], image.size[1]

        # 패딩 사이즈
        pad_h = int((target_size - h) / 2)
        h_rest = int((target_size - h) % 2) # 나머지
        pad_w = int((target_size - w) / 2)
        w_rest = int((target_size - w) % 2)
        
        # 사이즈 홀수라 나머지 있을 경우 아랫쪽, 오른쪽에 더해줌
        pad_size = (pad_h, pad_w, pad_h+h_rest, pad_w+w_rest)
        
        # 패딩
        return pad(image, pad_size, fill = 0)
    
    
    
    # png 이미지 넘파이 변경 후 채널 순서 변경
    # param
        # 이미지(PIL)
    # return : torch Tensor (c, h, w)
    def to_numpy_chw(self, image):
        np_arr = np.array(image.convert("RGB"))
        
        # (h, w, c) => (c, h, w) 채널 순서 변경
        np_arr = np.moveaxis(np_arr, 2, 0)
        
        return np_arr
    

    # 넘파이 리스트를 텐서로 바꿔줌 (메인에 torch import 안하려고 만듦)
    # param
        # 넘파이 어레이 (list)
    # return : tensor
    def to_tensor(self, numpy_arr):
        return torch.Tensor(numpy_arr)


    # 라벨 인덱싱
    # param
        # 라벨 리스트(list)
    # return : 인덱싱된 최종 라벨 (numpy_arr)
    def label_indexing(self, label_arr):
        
        label_kind = sorted(list(set(label_arr)))        
        
        # 클래스 변수 딕셔너리에 인덱스 저장
        for i, kind in enumerate(label_kind):
            self.__label_dic[kind] = i
        
        # 인덱싱 진행
        indexed_label_arr = []
        for label in label_arr:
            indexed_label_arr.append(self.__label_dic[label])
        
        # 갯수 검증 - 길이 다르면 에러 발생
        if len(label_arr) != len(indexed_label_arr):
            raise Exception
        
        # 넘파이로 변환 후 리턴        
        return np.array(indexed_label_arr)
        
    
    # 전체 이미지 변환 (리사이즈, 패딩, 넘파이 어레이/채널순서 변환)
    # param
        # 이미지 파일 경로 리스트 (list)
    # return : 최종 텐서 (torch Tensor)
    def make_tensor_arr(self, file_path, img_size):
        trans_image_arr = []
 
        for file in file_path:
            f = self.resize_image(file, img_size) # 리사이즈
            f = self.padding_image(f, img_size) # 패딩
            np_arr = self.to_numpy_chw(f) # 채널 순서 변경 후 넘파이 어레이 변환
            np_arr = np_arr / 255 # 0 ~ 1 사이로 스케일링
            np_arr = np.round(np_arr, 4)
            trans_image_arr.append(np_arr) # 한장 추가
        
        return torch.Tensor(np.array(trans_image_arr))

             
                
    # 두 리스트 원소들 쌍(path, label)을 하나의 튜플로 합쳐줌
    # param
        # path 리스트(list)
        # label 리스트(list)
    # return : 튜플 리스트(list)
    def sum_xy(self, lst_X, lst_Y):
        lst = []
        for path_lst, label_lst in zip(lst_X, lst_Y):
            for path, label in zip(path_lst, label_lst):
                lst.append((path, label))
        return lst


    # train, valid, test 타겟 비율 맞춰서 나누기
    # param
        # 파일 경로 리스트 (numpy_arr)
        # 인덱싱된 라벨 리스트 (numpy_arr)
        # 나눌 비율. test, valid 비율 (float)
    # return : test_set, valid_set, train_set (tuple)
    def split_test_vaild_train(self, file_path, indexed_label, ratio):

        file_path = np.array(file_path)
        
        # 튜플(path, label) 담을 리스트
        test_lst_X, test_lst_Y = [], []
        valid_lst_X, valid_lst_Y = [], []
        train_lst_X, train_lst_Y = [], []

        for t in list(self.get_label_dic().values()):
        
            # 해당 라벨(t) 인덱스 리스트. 튜플반환이라 0번 인덱스 가져옴
            t_idx = np.where(indexed_label == t)[0]

            # 랜덤으로 섞어줌
            np.random.shuffle(t_idx)
            
            # 전체 길이 중 비율만큼 갯수 산정. 반올림해줌
            split = int(round(len(t_idx) * ratio, 0))

            # test, valid, train 인덱스 나누기
            test_idx = t_idx[:split+1] # latio
            valid_idx = t_idx[split+1 : (split+1)*2] # latio
            train_idx = t_idx[(split+1)*2:] # 나머지
            
            # 각 인덱스에 따른 데이터 넣어주기
            test_lst_X.append(file_path[test_idx])
            test_lst_Y.append(indexed_label[test_idx])
            valid_lst_X.append(file_path[valid_idx])
            valid_lst_Y.append(indexed_label[valid_idx])
            train_lst_X.append(file_path[train_idx])
            train_lst_Y.append(indexed_label[train_idx])
        
        
        # (path, label) 튜플로 묶어서 최종 리스트 생성
        test_set = self.sum_xy(test_lst_X, test_lst_Y)
        valid_set = self.sum_xy(valid_lst_X, valid_lst_Y)
        train_set = self.sum_xy(train_lst_X, train_lst_Y) 
        
        return test_set, valid_set, train_set           
                
             
    # gpu, cpu 선택
    # return : device 종류 (str)
    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 랜덤 시드 고정
        torch.manual_seed(777)

        # GPU 사용 가능일 경우 랜덤 시드 고정
        if device == 'cuda':
            torch.cuda.manual_seed_all(777)

        return device  

        
    # 랜덤으로 섞고 배치 사이즈별로 X, Y 분리해서 리스트 생성 
    # param
        # 나눌 데이터 셋(list(tuple(path, label)))
        # 배치 사이즈 (int)
    # return : X(path), Y(label) 리스트
    def make_batch_arr(self, data_set, batch_size):

        # 데이터셋 랜덤으로 섞어줌
        shuffle(data_set)

        X, Y = [], [] # 최종 반환 리스트
        batch_x, batch_y = [], [] # 배치사이즈별로 담아줄 리스트
        
        for i, t in enumerate(data_set):
            
            # 배치사이즈에 도달하지 못하는 나머지는 버림
            if i == len(data_set) - len(data_set) % batch_size + 1:
                break

            # 배치 사이즈 다 차면 리스트에 담고 리셋
            if i > 0 and i % batch_size == 0:
                X.append(batch_x)
                Y.append(batch_y)
                
                # 다음 배치 담기 위해 초기화
                batch_x, batch_y = [], []

            # 배치 사이즈 다 차기 전에는 리스트에 하나씩 추가
            batch_x.append(t[0])
            batch_y.append(t[1])
        
        return X, Y

 
    # 데이터셋 X, Y  매칭 검증
    # param
        # X - path (list)
        # Y - path (list)
    # return : 일치여부 (bool 또는 key-error 발생)
    def verify_dataset(self, X, Y):
        
        label_dic = self.get_label_dic()
        isTrue = True
        
        for path_lst, label_lst in zip(X, Y):
            for path, label in zip(path_lst, label_lst):
                name = path.split("\\")[-2]
                
                # 두 폴더 이름 오타가 있어 넣어줌
                if name == "Gilt Head Bream":
                    name = "Gilt-Head Bream"
                elif name == "Horse Mackerel":
                    name = "Hourse Mackerel" 
                
                # 폴더 이름과 라벨이 일치하지 않으면 바로 종료
                if label != label_dic[name]:
                    isTrue = False
                    break
            
        return isTrue
    
              
    # 모델 저장
    # param
        # 모델 (moel.state_dict())
        # 저장 경로 (str)
    def save_model(self, model, path): 
        path = path 
        torch.save(model.state_dict(), path) 
    
    
    # 학습 진행
    # param
        # train 배치별 이미지 경로 (2차원 list)
        # train 인덱싱된 라벨 (1차원 list)
        # valid 배치별 이미지 경로 (2차원 list)
        # valid 인덱싱된 라벨 (1차원 list)
        # 에폭 횟수 (int)
        # gpu선택 디바이스 (str)
        # 모델 (torch model)
        # 러닝 레이트 보폭 (float)
        # 모델 저장 위치 (str)
    # return : 없음
    def run_epoch(self, train_X, train_Y, valid_X, valid_Y, training_epochs, device, model, learning_rate, img_size, save_path):
        
        criterion = torch.nn.CrossEntropyLoss().to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_batch = len(train_X)
        best_accuracy = 0
        
        # training            
        model.train() # 훈련모드 전환
        for epoch in range(training_epochs):
            
            avg_cost = 0

            # 미니 배치 단위로 꺼내오기
            for X, Y in zip(tqdm(train_X), train_Y): 
                
                X = self.make_tensor_arr(X, img_size)
                X = X.to(device)
                
                # 라벨은 long 타입만 받아줌
                Y = torch.Tensor(np.array(Y)).long()
                Y = Y.to(device)
                
                optimizer.zero_grad() # 그레디언트 초기화
                pred = model(X) # 모델 적용
                cost = criterion(pred, Y) # 손실함수 측정
                cost.backward() # 역전파
                optimizer.step() # 경사 하강
                avg_cost += cost / total_batch # 1 epoch당 평균 코스트
            
            # validation
            avg_valid_cost = 0
            size = len(valid_Y)
            # 개별 이미지 갯수
            # 배치로 묶여 있으면 batch_size * batch갯수)
            # 배치 사이즈와 갯수를 인자로 받지 않을 경우 별도로 계산필요
            image_count = 0 
            accuracy = 0 # 정확도 계산용
            with torch.no_grad():
                model.eval() # 검증모드 전환
                for x, y in zip(tqdm(valid_X), valid_Y):
                    x = self.make_tensor_arr(x, img_size)
                    x = x.to(device)
                    y = torch.Tensor(np.array(y)).long()
                    y = y.to(device)
                    
                    pred = model(x)
                    valid_cost = criterion(pred, y)
                    avg_valid_cost += valid_cost / size # 평균 코스트
                    
                    # 모델에서 나온 확률중 가장 높은 인덱스 추출
                    # max 값과 인덱스 튜플 반환
                    # valid를 배치 사이즈만큼 나눠서 넣는다면 배열로 나옴
                    pred_value, pred_index = torch.max(pred, 1)
                    
                    # y가 한번에 10개 들어오면 (10, size, size)이므로 size(0) = 10 (이미지 10장)
                    image_count += y.size(0)
                    
                    # 예측값과 실제값이 맞는 갯수를 추가
                    accuracy += (pred_index == y).sum().item()
            
            print(f'Epoch: {epoch + 1}, cost = {avg_cost:.9f}, valid_cost = {avg_valid_cost:.9f}')
            
            accuracy_ratio = accuracy / image_count
            print(f"validation = 일치 : {accuracy}/{image_count}, 정확도 : {accuracy_ratio:.9f}")   
            
            if accuracy_ratio * 100 > best_accuracy:
                self.save_model(model, save_path)
                best_accuracy = accuracy_ratio * 100
                print("모델 저장완료")
                
            
        
    # 세이브 모델 로딩 (파라미터만)
    # param
        # 모델 (해당 모델 객체)
        # 불러올 경로 (str)
    # return : 모델
    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model


    # 모델 최종 테스트
    # param
        # 테스트 이미지 경로 파일 리스트 (2차원 list)
        # 테스트 타겟 (1차원 list)
        # 모델
        # 이미지 사이즈
    def test_model(self, test_X, test_Y, device, model, img_size):
        
        model = model.to(device)
        accuracy = 0
        image_count = 0
        
        with torch.no_grad():
            for x, y in zip(tqdm(test_X), test_Y):
                x = self.make_tensor_arr(x, img_size)
                x = x.to(device)
                y = torch.Tensor(np.array(y)).long()
                y = y.to(device)
                
                pred = model(x)
                
                # 모델에서 나온 확률중 가장 높은 인덱스 추출
                # max 값과 인덱스 튜플 반환
                # test를 배치 사이즈만큼 나눠서 넣는다면 배열로 나옴
                pred_value, pred_index = torch.max(pred, 1)
                
                # y가 한번에 10개 들어오면 (10, size, size)이므로 size(0) = 10 (이미지 10장)
                image_count += y.size(0)
                
                # 예측값과 실제값이 맞는 갯수를 추가
                accuracy += (pred_index == y).sum().item()
        
        print(f"일치 : {accuracy} / {image_count}, {accuracy / image_count:.9f}") 
                

                
        
        
        
        
        
        
        
        
        
    