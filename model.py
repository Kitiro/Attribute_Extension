#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-06-10 09:48:32
LastEditTime: 2022-01-15 22:21:55
LastEditors: Kitiro
Description: 
FilePath: /Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/model.py
'''
from torch import nn
from torch.nn import functional as F
import torch

# class MyModel(nn.Module):
#     def __init__(self, attr_dim, output_dim):
#         super(MyModel, self).__init__()
#         self.attr_dim = attr_dim
#         self.output_dim = output_dim
#         self.conv_out_dim = 52 * (attr_dim + 2)

#         self.conv1 = nn.Conv2d(1, 50, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1)
#         self.fc = nn.Linear(self.conv_out_dim, self.output_dim)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  #

#         x = x.view(-1, 1, 50, self.attr_dim)  # batch, channel, height , width
#         x = F.relu(self.bn1(self.conv2(x))) # 50 x feature map

#         x = x.view(-1, self.conv_out_dim)  # flatten

#         x = self.fc(x)

#         return x

class MyModel(nn.Module):
    def __init__(self, attr_dim, output_dim):
        super(MyModel, self).__init__()
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        
        expand_dim = 50
        self.conv1 = nn.Conv2d(1, expand_dim, kernel_size=1)
        kernel_size = 3
        padding = 0
        self.conv2 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.conv_out_dim = (expand_dim+2*padding-kernel_size+1)*(attr_dim+2*padding-kernel_size+1)
        self.fc1 = nn.Linear(self.conv_out_dim, 4096)  # 
        self.fc2 = nn.Linear(4096, self.output_dim)
    def forward(self, x):

        x = F.relu(self.conv1(x)) # 
        x = x.view(-1, 1, 50, self.attr_dim)  # batch, channel, height , width
        x = F.relu(self.conv2(x))
        #x = x.view(-1, 50, 50, self.attr_dim)  
        #x = torch.mean(x, dim=1) # average the feature map
        x = x.view(-1, self.conv_out_dim)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x    

class APYModel(nn.Module):
    def __init__(self, attr_dim, output_dim):
        super(APYModel, self).__init__()
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        self.conv_out_dim = 48 * (attr_dim-2)

        self.conv1 = nn.Conv2d(1, 50, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.fc = nn.Linear(self.conv_out_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  #

        x = x.view(-1, 1, 50, self.attr_dim)  # batch, channel, height , width
        x = F.relu(self.bn1(self.conv2(x))) # 50 x feature map
        #x = torch.mean(x, dim=0)  # average the feature map  
        x = x.view(-1, self.conv_out_dim)  # flatten

        x = self.fc(x)

        return x