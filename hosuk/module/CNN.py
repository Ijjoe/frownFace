# CNN 모델 커스텀
import torch.nn.init

class CNN(torch.nn.Module):

    def __init__(self, img_size):
        
        # 이미지 사이즈
        self.img_size = img_size
        super(CNN, self).__init__()
        
        # conolution층
        self.conv = torch.nn.Sequential(
            
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32), # 배치 정규화
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 112
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 56
        
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 28
          
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 14
            
            torch.nn.Conv2d(256, 518, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2) # 7
            )


        # 전결합층
        self.header = torch.nn.Sequential(
            
            torch.nn.Linear(518 * 7 * 7, 5000, bias = True),
            torch.nn.ReLU(),
            
            torch.nn.Linear(5000, 1000, bias = True),
            torch.nn.ReLU(),
           
            torch.nn.Linear(1000, 50, bias = True),
            torch.nn.ReLU()
        )
        
        
        # 아홉번째 층 아웃풋 레이어
        self.fc = torch.nn.Linear(50, 2, bias=True)


        # 아웃풋 레이어 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)


    # 레이어 연결
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1) # 전결합층을 위해서 Flatten
        out = self.header(out)
        out = self.fc(out)
        return out
    
    