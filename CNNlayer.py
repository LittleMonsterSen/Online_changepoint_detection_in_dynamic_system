#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:29:18 2023

@author: senlin
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),          # B, 16 x 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),           # 8 x 8                
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),           # 4 x 4                 
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          #2 x 2
            nn.ReLU(True),
            nn.Conv2d(64, 512, 2),              # 1 x 1
            nn.Conv2d(512, output_dim, 1),            # B, z_dim*n parameter
        )

    def forward(self, x):
        h = x.view(-1, 1, 32, 32)
        z = self.encoder(h).view(x.size(0), self.output_dim)
        return z
    

class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, 1, 1, 0),     # 1 x 1 
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 64, 4, 1, 0),  # 4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 8 x 8 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 x 16  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),           
        )

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        mu_img = self.decoder(h)
        return mu_img