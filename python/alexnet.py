import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.maxp5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropf1 = nn.Dropout(p=0.5, inplace=True)
        self.fc1 = nn.Linear(in_features=(256 * 6 * 6), out_features=4096)
        self.reluf1 = nn.ReLU()
        self.dropf2 = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.reluf2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.maxp1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.maxp2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxp5(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropf1(x)
        x = self.fc1(x)
        x = self.reluf1(x)
        x = self.dropf2(x)
        x = self.fc2(x)
        x = self.reluf2(x)
        x = self.fc3(x)

        return x
    
    
def test():
    net = AlexNet()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()