import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 학습 진행 클래스
class RunCNN:

    # param : 데이터셋 (torch util dataset)
    def __init__(self):
        pass        
        
    # 데이터로더 만들기
    # param
        # 데이터셋(torch dataset)
        # 배치 사이즈(int)
        # 섞어주기 여부 (bool)
        # 남는 부분 버리기 여부 (bool)
    # return : 데이터로더 (torch util dataloader)
    def get_dataloader(self, dataset, batch_size, shuffle, drop_last):
        data_loader = DataLoader(dataset = dataset, 
                                 batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)
        return data_loader
    
    
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
    
    
    # 학습 진행
    # param
        # 에폭 횟수 (int)
        # train 데이터로더 (torch util dataloader)
        # valid 데이터로더 (torch util dataloader)
        # gpu선택 (str)
        # 모델 (torch model)
        # 러닝 레이트 보폭 (float)
    # return : 없음
    def run_epoch(self, training_epochs, train_data_loader, valid_data_loader, device, model, learning_rate, save_path):
        
        # 비용 함수에 소프트맥스 함수 포함되어져 있음.
        criterion = torch.nn.CrossEntropyLoss().to(device)    
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_batch = len(train_data_loader)
        best_accuracy = 0
        
        # training            
        model.train() # 훈련모드 전환
        for epoch in range(training_epochs):
            
            avg_cost = 0

            # X 미니 배치, Y 레이블.
            for X, Y in tqdm(train_data_loader): 

                X = X.to(device)
                Y = Y.to(device).long() # 롱타입만 받음

                optimizer.zero_grad()
                pred = model(X)
                cost = criterion(pred, Y)
                cost.backward()
                optimizer.step()

                avg_cost += cost / total_batch
                

            # validation
            avg_valid_cost = 0
            size = len(valid_data_loader)
            # 개별 이미지 갯수
            # 배치로 묶여 있으면 batch_size * batch갯수)
            # 배치 사이즈와 갯수를 인자로 받지 않을 경우 별도로 계산필요
            image_count = 0 
            accuracy = 0 # 정확도 계산용
            
            with torch.no_grad():
                model.eval() # 검증모드 전환
                for x, y in tqdm(valid_data_loader): 

                    x = x.to(device)
                    y = y.to(device).long() # 롱타입만 받음

                    pred = model(x)
                    valid_cost = criterion(pred, y)
                    
                    avg_valid_cost += valid_cost / size # 평균 코스트
                
                    # 모델에서 나온 확률중 가장 높은 인덱스 추출
                    # max 값과 인덱스 튜플 반환
                    # valid를 배치 사이즈만큼 나눠서 넣는다면 배열로 나옴
                    pred_value, pred_index = torch.max(pred, 1)
                    
                    # y가 한번에 10개 들어오면 (10, size, size)이므로 size(0) = 10
                    image_count += y.size(0)
                    
                    # 예측값과 실제값이 맞는 갯수를 추가
                    accuracy += (pred_index == y).sum().item()
                    
                    
            accuracy_ratio = accuracy / image_count
            print(f'Epoch: {epoch + 1}, cost = {avg_cost:.9f}, valid_cost = {avg_valid_cost:.9f}')
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
                