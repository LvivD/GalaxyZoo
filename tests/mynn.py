import torch
import torch.nn as nn

class MyCNN(nn.Module):
#     3x96 -5> 16x92 -> 46 -3> 32x44 -> 22 -3> 64x20 -> 10 -3> 128x8 -> 4
    def __init__(self):
        super(MyCNN, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_1 = conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        
        self.conv_2 = conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3 = conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.conv_4 = conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.hidden= nn.Sequential(
            nn.Linear(128*4*4, 128), 
            nn.ReLU(inplace=True)
        )
        
        self.drop = nn.Dropout(0.2)
        
        self.out = nn.Sequential(
            nn.Linear(128, 37), 
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, image):
        
        x1 = self.conv_1(image.float())
        x2 = self.max_pool_2x2(x1)
        x3 = self.conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = x8.view(x8.shape[0], -1)
        x10 = self.hidden(x9)
        x11 = self.drop(x10)
        x = self.out(x11) 
        return x
    
class MyCNNV2(nn.Module):
#     3x96 -5> 16x92 -> 46 -3> 32x44 -> 22 -3> 64x20 -> 10 -3> 128x8 -> 4
#   3x132 -5> 16x128 -> 64 -5> 32x60 -> 30 -3> 64x28 -> 14 -3> 128x12 -> 6 
    def __init__(self):
        super(MyCNN, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_1 = conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        
        self.conv_2 = conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3 = conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.conv_4 = conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.hidden= nn.Sequential(
            nn.Linear(128*6*6, 128), 
            nn.ReLU(inplace=True)
        )
        
        self.drop = nn.Dropout(0.2)
        
        self.out = nn.Sequential(
            nn.Linear(128, 37), 
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, image):
        
        x1 = self.conv_1(image.float())
        x2 = self.max_pool_2x2(x1)
        x3 = self.conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = x8.view(x8.shape[0], -1)
        x10 = self.hidden(x9)
        x11 = self.drop(x10)
        x = self.out(x11) 
        return x
        