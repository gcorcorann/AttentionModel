#!/usr/bin/env python3
"""
Attention Models

Author: Gary Corcoran
Date: Jan. 22nd, 2018
"""
import torch.nn as nn
import torchvision.models as models

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # reshape into [numSeqs x batchSize x numFeats]
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(0,1)
        output, hidden = self.rnn(x)
        output = self.linear(output)
        return output

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained ResNet-152 and replace top fc layer
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete last fc layer
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(resnet.fc.in_features, 10)
        
    def forward(self, x):
        """Extract the image feature vectors."""
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.resnet(x)
        return features
