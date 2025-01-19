import torch
import torch.nn as nn
from torchvision import models, transforms
import utils

class VGG(nn.Module):
    def __init__(self,vgg_19=True, vgg_path='None'):
        super().__init__()
        self.vgg_19 = vgg_19
        
        if self.vgg_19:
            vgg_features = models.vgg19(pretrained=False)
        else: 
            vgg_features = models.vgg16(pretrained=False)
        vgg_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg_features.features
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        if self.vgg_19:
            layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '22': 'relu4_2', '26': 'relu4_4', '35': 'relu5_4'}
        else:
            layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
            
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if (not self.vgg_19 and name == '22'): #or (self.vgg_19 and name == '35'):
                    break
        return features
    
