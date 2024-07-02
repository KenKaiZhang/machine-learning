import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_red, out_3x3, out_5x5_red, out_5x5, pool_proj):
        super().__init__()
        self.branch1 = Conv(in_channels, out_1x1, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            Conv(in_channels, out_3x3_red, kernel_size=1, stride=1),
            Conv(out_3x3_red, out_3x3, kernel_size=3, stride=1, padding=1), 
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, out_5x5_red, kernel_size=1, stride=1),
            Conv(out_5x5_red, out_5x5, kernel_size=5, stride=1, padding=2), 
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_channels, pool_proj, kernel_size=1, stride=1), 
        )


    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        x = [branch1, branch2, branch3, branch4]
        x = torch.cat(x, 1)
        return x
    

class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(5)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1)
        self.convR = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fcR = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.convR(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fcR(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    

class GoogLeNet(nn.Module):
    def __init__(self, aux, dropout, aux_dropout, num_classes):
        super().__init__()
        self.conv1 = Conv(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv(64, 64, kernel_size=1, stride=1)
        self.conv2a = Conv(64, 192, kernel_size=3, stride=1, padding=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxp3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 25, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxp4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = AuxClassifier(512, num_classes, aux_dropout) if aux else None
        self.aux2 = AuxClassifier(528, num_classes, aux_dropout) if aux else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)

        x = self.conv2(x)
        x = self.conv2a(x)
        x = self.maxp2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxp3(x)

        x = self.inception4a(x)

        y = self.aux1(x) if self.aux1 and self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        z = self.aux2(x) if self.aux2 and self.training else None

        x = self.inception4e(x)
        x = self.maxp4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        if self.aux1 and self.aux2 and self.training:
            return x, y, z
        return x
    

def test():
    net = GoogLeNet(False, 0.4, 0.7, 1000)
    x = torch.randn(10, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()