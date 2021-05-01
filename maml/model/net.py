from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d
from torch.nn.modules.pooling import MaxPool2d
import torchvision.models as models

from utils.config import _C as C


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, ks, padding=0, stride=1, norm=True, act=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, ks, stride, padding),
            nn.BatchNorm2d(out_features) if norm else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=True, act=True, dropout=0.3):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features) if norm else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity(),
            nn.Dropout(dropout) if act else nn.Identity() 
        )
    
    def forward(self, x):
        return self.block(x)

def change_params(net, new):
        for n,v in new:
            for i in range(11):
                n = n.replace(f".{i}.", f"[{i}].")
            exec(f'net.{n}.data = v')

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 32)

    def forward(self, x, new_params=None):
        if new_params is not None:
            old_params = deepcopy(list(self.named_parameters()))
            change_params(new_params)
        
        out = self.backbone(x)
        
        if new_params is not None:
            change_params(old_params)
        return out

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 32, 3),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 32, 3),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 32, 3),
            MaxPool2d(2, 1),
            nn.Flatten(),
            nn.Linear(32*5*5, C.MAML.N_WAY)
        )
    
    def forward(self, x, new_params=None):
        if new_params is not None:
            old_params = deepcopy(list(self.named_parameters()))
            change_params(self, new_params)
        
        out = self.backbone(x)
        
        if new_params is not None:
            change_params(self, old_params)
        return out



class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.l1 = nn.Linear(5, 1, bias=False)

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
            n = n.replace(".0.", "[0].")
            n = n.replace(".1.", "[1].")
            exec(f'self.{n}.data = v')

def toy_test():
    model = Test()
    x1 = torch.randn(2, 5)
    x2 = torch.randn(2, 5)

    
    print("Original model weight:", model.l1.weight.data)

    optim = torch.optim.Adam(model.parameters(), lr=1e-1)

    out_1_init = model(x1)
    loss = torch.mean((out_1_init - 1) ** 2)
    overall_loss = loss
    grad = torch.autograd.grad(loss, model.parameters())
    fast_weights = list(map(lambda p: (p[1][0], p[1][1] - 0.001 * p[0]), zip(grad, model.named_parameters())))
    print("new1 weights:", fast_weights[0][1])
    out1 = model(x1, fast_weights)
    loss1 = torch.mean((out1 - 1) ** 2)

    print("out_1_init:", out_1_init.detach().numpy().flatten())
    print("out1:", out1.detach().numpy().flatten())
    print(model.l1.weight.data)

    loss = torch.mean((model(x2) - 2) ** 2)
    grad = torch.autograd.grad(loss, model.parameters())
    fast_weights = list(map(lambda p: (p[1][0], p[1][1] - 0.001 * p[0]), zip(grad, model.named_parameters())))

    print("new2 weights:", fast_weights[0][1])
    out2 = model(x2, fast_weights)
    loss2 = torch.mean((out2 - 1) ** 2)

    print("out2:", out2.detach().numpy().flatten())
    print(model.l1.weight.data)
    
    overall_loss = loss1 + loss2
    overall_loss.backward()
    optim.step()

    print("New model weight:", model.l1.weight.data)

def net_test():
    model = Net()
    model.change_params(model.named_parameters())


if __name__ == '__main__':
    torch.manual_seed(0)
    toy_test()
    # net_test()


    

