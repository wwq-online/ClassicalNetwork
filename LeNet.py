import torch
from torch import nn
from torch.nn import Module

"""
LeNet-5
"""


class LeNet(Module):
    def __init__(self, in_channels, num_class):
        super(LeNet, self).__init__()
        self.front_end = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0),     # 32x32->28x28
            nn.ReLU(True),      # to save memory, inplace was set to True,
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28->14x14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 14x14->10x10
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 10x10->5x5
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_class)
        )

    def forward(self, x):
        out = self.front_end(x)
        # flatten
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


num_class = 10
b, c, h, w = 2, 1, 32, 32
model = LeNet(c, num_class)
img = torch.ones(b, c, h, w)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using cuda ......')
    model = model.cuda()
    img = img.cuda()

out = model(img)
print(out.shape)



