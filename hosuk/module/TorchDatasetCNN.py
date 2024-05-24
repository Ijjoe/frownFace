import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# CNN 구동 클래스
class TorchDatasetCNN:
    
    def __init__(self):
        
        # 객체 생성과 동시에 랜덤시드 고정
        self.generator = torch.Generator().manual_seed(777)
    

    # datset 비율대로 나누기
    # param
        # 전체 데이터셋(torch dataset)
        # 나눌 비율 (list(나눌 갯수만큼))
    def split_dataset(self, dataset, ratio):
        return random_split(dataset, ratio, generator=self.generator)
     

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
    
    
    # 모델 저장
    # param
        # 모델 (moel.state_dict())
        # 저장 경로 (str)
    def save_model(self, model, path): 
        path = path 
        torch.save(model.state_dict(), path) 
    
    
    # 학습 진행
    # param
        # 에폭 횟수 (int)
        # train 데이터로더 (torch util dataloader)
        # valid 데이터로더 (torch util dataloader)
        # 디바이스 선택 (str)
        # 모델 (torch model)
        # 러닝 레이트 보폭 (float)
    # return : train 정확도(list), valid 정확도(list), valid 리콜(list)
    def run_epoch(self, training_epochs, train_dataloader, valid_dataloader, device, model, learning_rate, save_path):
        
        # 비용 함수에 소프트맥스 함수 포함되어져 있음.
        criterion = torch.nn.CrossEntropyLoss().to(device)    
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        
        total_batch = len(train_dataloader)
        train_image_count = total_batch * train_dataloader.batch_size # 전체 이미지 갯수
        best_recall = 0
        train_acc, valid_acc, valid_recall = [], [], [] # 에폭마다 저장 (그래프용)
         
        # training            
        model.train() # 훈련모드 전환
        for epoch in range(training_epochs):
            
            avg_cost = 0 # 에폭 당 코스트 계산용
            train_accuracy = 0 # 에폭 당 정확도 계산용
            tp_sum, fn_sum = 0, 0 # 에폭 당 recall 계산용

            # X 미니 배치, Y 레이블.
            for X, Y in tqdm(train_dataloader): 

                X = X.to(device)
                Y = Y.to(device).long() # 롱타입만 받음

                optimizer.zero_grad()
                pred = model(X)
                cost = criterion(pred, Y)
                cost.backward()
                optimizer.step()

                avg_cost += cost / total_batch

                # 모델에서 나온 확률중 가장 높은 인덱스 추출
                # max 값과 인덱스 튜플 반환
                # valid를 배치 사이즈만큼 나눠서 넣는다면 배열로 나옴
                pred_value, pred_index = torch.max(pred, 1)
                
                # 예측값과 실제값이 맞는 갯수를 추가
                train_accuracy += (pred_index == Y).sum().item()
                

            # validation
            avg_valid_cost = 0
            size = len(valid_dataloader)
            
            # 개별 이미지 갯수
            # 배치로 묶여 있으면 batch_size * batch갯수)
            # 배치 사이즈와 갯수를 인자로 받지 않을 경우 별도로 계산필요
            valid_image_count = 0 
            valid_accuracy = 0 # 정확도 계산용
            
            with torch.no_grad():
                model.eval() # 검증모드 전환
                for x, y in tqdm(valid_dataloader): 

                    x = x.to(device)
                    y = y.to(device).long() # 롱타입만 받음

                    pred = model(x)
                    valid_cost = criterion(pred, y)
                    
                    avg_valid_cost += valid_cost / size # 평균 코스트
                
                    pred_value, pred_index = torch.max(pred, 1)
                    valid_image_count += y.size(0)
                    valid_accuracy += (pred_index == y).sum().item()
                    
                    # 실제 산불(1) 중 예측이 맞는것(tp)과 틀린것(fn) - racall 계산용
                    tp_sum += sum([1 for item1, item2 in zip(y, pred_index) if item1 == 1 and item1 == item2])
                    fn_sum += sum([1 for item1, item2 in zip(y, pred_index) if item1 == 1 and item1 != item2])       
                    
            train_accuracy_ratio = train_accuracy / train_image_count * 100
            valid_accuracy_ratio = valid_accuracy / valid_image_count * 100
            print(f'Epoch: {epoch + 1}, cost = {avg_cost:.9f}, valid_cost = {avg_valid_cost:.9f}')
            print(f"train = 일치 : {train_accuracy}/{train_image_count}, 정확도 : {train_accuracy_ratio:.2f}")   
            print(f"validation = 일치 : {valid_accuracy}/{valid_image_count}, 정확도 : {valid_accuracy_ratio:.2f}")   
            
            # 에폭 당 정확도 저장
            train_acc.append(train_accuracy_ratio)
            valid_acc.append(valid_accuracy_ratio)
                        
            # validation Recall 계산
            recall = tp_sum / (tp_sum + fn_sum) * 100
            valid_recall.append(recall)
            print(f"리콜 = tp : {tp_sum}, fn : {fn_sum} / racall : {recall:.2f}")
            
            # 리콜 기준으로 모델 저장
            if recall > best_recall:
                self.save_model(model, save_path)
                best_recall = recall
                print("모델 저장완료")

        # 결과 그래프 그리기                
        self.make_result_graph(train_acc, valid_acc, valid_recall)
        
        return train_acc, valid_acc, valid_recall
                
                
    # 세이브 모델 로딩 (파라미터만)
    # param
        # 모델 (해당 모델 객체)
        # 불러올 경로 (str)
        # 디바이스 정보 (str)
    # return : 모델
    def load_model(self, model, path, device):
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        return model


    # 모델 최종 테스트
    # param
        # 테스트 데이터로더 (torch util dataloader)
        # 모델
        # 이미지 사이즈
    def test_model(self, test_dataloader, device, model):
        
        model = model.to(device)
        test_accuracy = 0
        image_count = 0
        tp_sum, fn_sum = 0, 0
        
        with torch.no_grad():
            model.eval()
            for x, y in tqdm(test_dataloader):

                x = x.to(device)
                y = y.to(device).long()
                
                pred = model(x)
                
                # 모델에서 나온 확률중 가장 높은 인덱스 추출
                # max 값과 인덱스 튜플 반환
                # test를 배치 사이즈만큼 나눠서 넣는다면 배열로 나옴
                pred_value, pred_index = torch.max(pred, 1)
                
                # y가 한번에 10개 들어오면 (10, size, size)이므로 size(0) = 10 (이미지 10장)
                image_count += y.size(0)
                
                # 예측값과 실제값이 맞는 갯수를 추가
                test_accuracy += (pred_index == y).sum().item()
                
                # 실제 산불(1) 중 예측이 맞는것(tp)과 틀린것(fn) - racall 계산용
                tp_sum += sum([1 for item1, item2 in zip(y, pred_index) if item1 == 1 and item1 == item2])
                fn_sum += sum([1 for item1, item2 in zip(y, pred_index) if item1 == 1 and item1 != item2])       
   
   
        # 정확도 출력        
        print(f"일치 : {test_accuracy} / {image_count}, 정확도 : {test_accuracy / image_count * 100:.2f}") 
 
        # Recall 계산
        test_recall = tp_sum / (tp_sum + fn_sum) * 100
        print(f"리콜 = tp : {tp_sum}, fn : {fn_sum} / racall : {test_recall:.2f}")
        
        
    # 앙상블 테스트
    # param
        # 모델 리스트 (list)
        # 데스트 데이터로더 (torch util dataloader)
        # 디바이스 (str)
    # return : 없음
    def test_ensenble_model(self, model_lst, test_dataloader, device, kind):        
        test_accuracy = 0
        image_count = 0
        tp_sum, fn_sum = 0, 0
        
        model_count = len(model_lst) # 모델 갯수

        with torch.no_grad():

            model_eval = [model.eval() for model in model_lst] # 평가모드 전환
            
            for x, y in tqdm(test_dataloader):

                x = x.to(device)
                y = y.to(device).long()

                # 각 모델에서 나오는 예측값
                pred_lst = []
                for model in model_eval:
                    pred_lst.append(model(x))
                    
                
                # 모든 모델의 평균 예측값
                ensenble_pred = sum(pred_lst) / model_count

                # 확률값 중 높은 인덱스(예측값 0 vs 1) 추출
                _, ensenble_pred_index = torch.max(ensenble_pred, 1)

                # 배치가 한번에 20개 들어오면 (20, 3, size, size)이므로 size(0) = 20 (이미지 20장)
                image_count += y.size(0)

                # 예측값과 실제값이 맞는 갯수를 추가
                test_accuracy += (ensenble_pred_index == y).sum().item()

                # 실제 산불(1) 중 예측이 맞는것(tp)과 틀린것(fn) - racall 계산용
                tp_sum += sum([1 for item1, item2 in zip(y, ensenble_pred_index) if item1 == 1 and item1 == item2])
                fn_sum += sum([1 for item1, item2 in zip(y, ensenble_pred_index) if item1 == 1 and item1 != item2])       

        # 종류 출력
        print(kind)
        
        # 정확도 출력        
        print(f"일치 : {test_accuracy} / {image_count}, 정확도 : {test_accuracy / image_count * 100:.2f}") 

        # Recall 계산
        test_recall = tp_sum / (tp_sum + fn_sum) * 100
        print(f"리콜 = tp : {tp_sum}, fn : {fn_sum} / racall : {test_recall:.2f}")        
        
        
        
    # 결과 그래프 그리기
    # param
        # train 모든 에폭 정확도 (list)
        # valid 모든 에폭 정확도 (list)
        # valid 모든 에폭 리콜 (list)
    # return : 그래프 출력
    def make_result_graph(self, train_acc, valid_acc, valid_recall):
        
        plt.figure(figsize = (10,3))
        
        # acc 그래프
        idx = [i+1 for i in range(len(train_acc))] # X축 (에폭 횟수)
        plt.subplot(1,2,1)
        plt.plot(idx, train_acc, color = "red", marker = "o", alpha = 0.5, linewidth = 2)
        plt.plot(idx, valid_acc, color = "green", marker = "x", alpha = 0.5, linewidth = 2)
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("accuracy")
        plt.legend(('Train_acc','Valid_acc'))

        # recall 그래프
        plt.subplot(1,2,2)
        plt.plot(idx, valid_recall, color = "red", marker = "o", alpha = 0.5, linewidth = 2)
        plt.title("Recall")
        plt.xlabel("Epoch")
        plt.ylabel("recall")

        plt.show()
        
