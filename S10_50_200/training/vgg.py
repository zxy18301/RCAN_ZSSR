# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:17:19 2023

@author: 93969
"""


import torch
import torch.nn as nn
import torchvision.models as models

def normalize_tensor_batch(tensor):
    # Get the minimum and maximum values for each batch sample
    min_val = tensor.min(dim=(2, 3), keepdim=True)[0]
    max_val = tensor.max(dim=(2, 3), keepdim=True)[0]

    # Normalize the tensor between 0 and 1 for each batch sample
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor

# Define the VGG loss function
class VGGLoss(nn.Module):
    def __init__(self):
        
        super(VGGLoss, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        # if conv_index == '22':
        self.vgg = nn.Sequential(*modules[:8])
        # elif conv_index == '54':
            # self.vgg = nn.Sequential(*modules[:35])
        
        self.vgg.requires_grad = False
        self.criterion = nn.MSELoss()
        
        
        
        
        
        
        
        
        # super(VGGLoss, self).__init__()
        
        # # Load the pre-trained VGG network
        # vgg = models.vgg19(pretrained=True).features
        # vgg = nn.Sequential(*list(vgg.children())[:35]).eval()
        
        # # Freeze the parameters of the VGG network
        # for param in vgg.parameters():
        #     param.requires_grad = False
        
        # self.vgg = vgg
        # self.criterion = nn.MSELoss()

    def forward(self, inputs, target):
        
        if inputs.shape[1] == 1:  # Convert grayscale to 3 channels
            inputs = inputs.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
    
    
        # inputs = (inputs-0.437) / 0.225
        # target = (target-0.437) / 0.225
        input_features = self.vgg(inputs)
        target_features = self.vgg(target)
        loss = self.criterion(input_features, target_features)
        return loss