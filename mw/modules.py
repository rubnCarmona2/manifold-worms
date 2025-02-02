import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualModule(nn.Module):
    def __init__(self, dims):
        super(ResidualModule, self).__init__()
        self.linear = nn.Linear(dims, dims)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x_1 = self.linear(x)
        x_1 = F.relu(x_1)
        x = x + x_1
        return x


class ReluNeuron(nn.Module):
    def __init__(self, dims):
        super(ReluNeuron, self).__init__()
        self.linear = nn.Linear(dims, dims)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = F.relu(x)
        x = self.linear(x)
        return x
