import torch
import torch.nn as nn
import torchvision.models as models

from copy import deepcopy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 32)

    def forward(self, x):
        return self.backbone(x)


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.l1 = nn.Linear(5, 10, bias=False)

    def forward(self, x, new_params=None):
        if new_params is not None:
            old_params = deepcopy(list(self.named_parameters()))
            self.change_params(new_params)
        out = self.l1(x)
        if new_params is not None:
            self.change_params(old_params)
        return out

    def change_params(self, new):
        for n,v in new:
            exec(f'self.{n} = v')


if __name__ == '__main__':
    torch.manual_seed(0)
    model = Test()
    x = torch.randn(2, 5)
    params = list(model.named_parameters())
    print(params[0][0])

    new_params = list(map(lambda p: (p[0], torch.nn.parameter.Parameter(torch.zeros(10,5))), params))
    print(new_params[0])

    # print(list(model.named_parameters()))
    print(model(x))
    print(model(x, new_params))
    print(model(x))

