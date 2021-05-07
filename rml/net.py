import torch 
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class RML(nn.Module):
    def __init__(self):
        super(RML, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 5)
    
    def forward(self, x):
        out = self.model(x)
        return nn.Sigmoid()(out)

    def extract_feature(self, x):
        input_batch = x.unsqueeze(0)

        with torch.no_grad():
            my_output = None
            
            def my_hook(module_, input_, output_):
                nonlocal my_output
                my_output = output_

            a_hook = self.model.layer4.register_forward_hook(my_hook)        
            self.model(input_batch)
            a_hook.remove()
            return my_output