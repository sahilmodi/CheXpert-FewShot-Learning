import torch
import torch.nn as nn
import torchvision.models as models

from copy import deepcopy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 32)

    def forward(self, x, new_params=None):
        if new_params is not None:
            old_params = deepcopy(list(self.named_parameters()))
            self.change_params(new_params)
        
        out = self.backbone(x)
        
        if new_params is not None:
            self.change_params(old_params)
        return out

    def change_params(self, new):
        for n,v in new:
            n = n.replace(".0.", "[0].")
            n = n.replace(".1.", "[1].")
            exec(f'self.{n}.data = v')


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

    
    print(model.l1.weight.data)

    optim = torch.optim.Adam(model.parameters(), lr=1e-1)

    loss = torch.mean((model(x1) - 1) ** 2)
    overall_loss = loss
    grad = torch.autograd.grad(loss, model.parameters())
    fast_weights = list(map(lambda p: (p[1][0], p[1][1] - 0.001 * p[0]), zip(grad, model.named_parameters())))
    print("new1 weights:", fast_weights[0][1])
    out1 = model(x1, fast_weights)
    loss1 = torch.mean((out1 - 1) ** 2)

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

    print(model.l1.weight.data)

def net_test():
    model = Net()
    model.change_params(model.named_parameters())


if __name__ == '__main__':
    torch.manual_seed(0)
    # toy_test()
    # net_test()


    

