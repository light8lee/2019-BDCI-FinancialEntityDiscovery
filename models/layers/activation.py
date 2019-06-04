import numpy as np
import torch as t
import torch.nn as nn
from torch.nn.modules.activation import *

def gelu(x):
    cdf = 0.5 * (1.0 + t.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * t.pow(x, 3))
    ))
    return x * cdf


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return gelu(x)