import torch
from torch import nn
from torch.nn import Module

"""
AlexNet-5(deeper):
    - 5 Conv (+ ReLU + Pooling)
    - 3 FC ( + ReLU), include classifier FC
"""


class AlexNet(Module):
    def __init__(self, num_class):
        super(AlexNet, self).__init__()
        self.front_end = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),     # 227x227 -> 55x55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   # 55x55 -> 27x27
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),    # 27x27 -> 27x27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   # 27x27 -> 13x13
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),   # 13x13 -> 13x13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # 13x13 -> 13x13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   # 13x13 -> 6x6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),      # 256x6x6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        x = self.front_end(x)
        # flatten
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


num_class = 1000
b, c, h, w = 2, 3, 227, 227
model = AlexNet(num_class)
img = torch.ones(b, c, h, w)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using cuda ......')
    model = model.cuda()
    img = img.cuda()

out = model(img)
print(out.shape)



