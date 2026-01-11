import torchvision
from torch import nn


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=False)
        self.f9 = nn.Linear(1000, 2)

    # 这个 forward 函数名是固定不可更改的
    def forward(self, x):
        x = self.model(x)
        x = self.f9(x)
        return x
