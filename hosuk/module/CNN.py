# CNN 모델 커스텀
import torch.nn.init

class CNN(torch.nn.Module):

    def __init__(self, img_size):
        
        # 이미지 사이즈
        self.img_size = img_size
        super(CNN, self).__init__()
        
        
        # 첫번째층 (Conv)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            # 풀링
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
            
        img_size = int(img_size/2) # 풀링 후 이미지 사이즈


        # 두번째층 (Conv)
        self.layer2 = torch.nn.Sequential(
        
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        img_size = int(img_size/2) # 풀링 후 이미지 사이즈
        
        
        # 세번째층 (Conv)
        self.layer3 = torch.nn.Sequential(
        
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        img_size = int(img_size/2) # 풀링 후 이미지 사이즈
        
        
        # 네번째층 (Conv)
        self.layer4 = torch.nn.Sequential(
        
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        img_size = int(img_size/2) # 풀링 후 이미지 사이즈
        
        
        # 다섯번째층 (Conv)
        self.layer5 = torch.nn.Sequential(
        
            torch.nn.Conv2d(256, 518, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        img_size = int(img_size/2) # 풀링 후 이미지 사이즈


        # 여섯 번째 층 (FC)
        self.layer6 = torch.nn.Sequential(
            
            torch.nn.Linear(518 * img_size * img_size, 5000, bias = True),
            torch.nn.ReLU()
        )
        
        
        # 일곱번째 층 (FC)
        self.layer7 = torch.nn.Sequential(
            
            torch.nn.Linear(5000, 1000, bias = True),
            torch.nn.ReLU()
        )
        
        
        # 여덟번째 층 (FC)
        self.layer8 = torch.nn.Sequential(
            
            torch.nn.Linear(1000, 50, bias = True),
            torch.nn.ReLU()
        )


        # 아홉번째 층 아웃풋 레이어
        self.fc = torch.nn.Linear(50, 2, bias=True)


        # 아웃풋 레이어 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)


    # 레이어 연결
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1) # 전결합층을 위해서 Flatten
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.fc(out)
        return out