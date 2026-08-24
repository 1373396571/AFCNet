import torch.nn as nn
class ADD(nn.Module):
    def __init__(self, arg):
        super(ADD, self).__init__()
        self.arg = arg
    def forward(self, x):
        return x[1] + x[0]