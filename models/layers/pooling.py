import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math

class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, input):
        """[summary]
        
        Arguments:
            input [b, t, h] -- [description]
        
        Returns:
            [b, h] -- [description]
        """
        assert len(input.shape) == 3
        output, _ = torch.max(input, 1)
        return output


class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, input):
        """[summary]
        
        Arguments:
            input [b, t, h] -- [description]
        
        Returns:
            [b, h] -- [description]
        """
        assert len(input.shape) == 3
        output = torch.mean(input, 1)
        return output


class GlobalSumPooling(nn.Module):
    def __init__(self):
        super(GlobalSumPooling, self).__init__()

    def forward(self, input):
        """[summary]
        
        Arguments:
            input [b, t, h] -- [description]
        
        Returns:
            [b, h] -- [description]
        """
        assert len(input.shape) == 3
        output = torch.sum(input, 1)
        return output