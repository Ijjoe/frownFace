# CNN 모델 커스텀
import torch.nn.init

class CNN(torch.nn.Module):

    def __init__(self):
        
        super(CNN, self).__init__()
        
        # 첫번째층
        # 100, 3, 256, 256
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            # 풀링
            # 100, 32, 128, 128
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        # 두번째층
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            # 100, 64, 64, 64 
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 64x64x64 inputs -> 9 outputs
        self.fc = torch.nn.Linear(64 * 64 * 64, 9, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)


    # 레이어 연결
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out