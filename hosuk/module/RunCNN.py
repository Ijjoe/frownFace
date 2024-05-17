import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    def run_epoch(self, training_epochs, train_data_loader, valid_data_loader, device, model, learning_rate):
        
        # 비용 함수에 소프트맥스 함수 포함되어져 있음.
        criterion = torch.nn.CrossEntropyLoss().to(device)    
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_batch = len(train_data_loader)
        
        for epoch in range(training_epochs):
            avg_cost = 0

            # X 미니 배치, Y 레이블.
            for X, Y in tqdm(train_data_loader): 

                X = X.to(device)
                Y = Y.to(device).long() # 롱타입만 받음
                
                

                optimizer.zero_grad()
                hypothesis = model(X)
                cost = criterion(hypothesis, Y)
                cost.backward()
                optimizer.step()

                avg_cost += cost / total_batch
                

            # validation
            avg_valid_cost = 0
            size = len(valid_data_loader)
            
            with torch.no_grad():

                for x, y in tqdm(valid_data_loader): 

                    x = x.to(device)
                    y = y.to(device).long() # 롱타입만 받음

                    hypothesis = model(x)
                    cost = criterion(hypothesis, y)
                    
                    avg_valid_cost += cost / total_batch
                

            print('[Epoch: {:>1}] cost = {:>.9}, valid_cost = {:>.9}'.format(epoch + 1, avg_cost, avg_valid_cost))
            
