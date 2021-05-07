import torch 
import torch.nn as nn
from torchvision import models

class RML(nn.Module):
    def __init__(self):
        super(RML, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 5)
    
    def forward(self, x):
        out = self.model(x)
        return nn.Sigmoid()(out)