import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, input, axis=1):
        """[summary]
        
        Arguments:
            input [b, t, h] -- [description]
        
        Returns:
            [b, h] -- [description]
        """
        output, _ = torch.max(input, axis)
        return output


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, input, axis=1):
        """[summary]
        
        Arguments:
        
        Returns:
            [b, h] -- [description]
        """
        output = torch.mean(input, axis)
        return output


class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, input, axis=1):
        """[summary]
        
        Arguments:
            input [b, t, h] -- [description]
        
        Returns:
            [b, h] -- [description]
        """
        output = torch.sum(input, axis)
        return output