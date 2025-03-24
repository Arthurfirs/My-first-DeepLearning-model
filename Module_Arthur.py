
from torch import nn

# 搭建神经网络
class Arthur(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32,32,5,1,2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256,10)
        )


    def forward(self,x):
        x = self.module(x)
        return x