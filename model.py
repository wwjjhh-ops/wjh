import torch
from torch import nn


class VGG16_BN(nn.Module):

    def __init__(self):

        super(VGG16_BN, self).__init__()

        self.model1 = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.model2 = nn.Sequential(

            nn.Linear(512, 64),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.model1(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.model2(x)

        return x