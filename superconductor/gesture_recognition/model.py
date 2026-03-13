import torch
from torch import nn

class PalmModel(nn.Module):
    def __init__(self, input_features=63*2, num_classes=2):
        super(PalmModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_features, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)

        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        z1 = self.fc1(x)
        h1 = self.relu(z1)
        
        z2 = self.fc2(h1)
        h2 = self.relu(z2)
        
        z3 = self.fc3(h2)
        
        return z3

