import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1 input 32/1/1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            # nn.Dropout(dropout_value)
        ) # output_size = 32/3

        #layer1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 32/5

        self.resnet_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 32/5


        # layer 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 16/5


        # layer 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 8/3

        self.resnet_block2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 16/5

        self.pool3 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(512, 10)



        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + self.resnet_block1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x + self.resnet_block2(x)
        x = self.pool3(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=-1)
        # print(x.shape)
        return x

    
    
