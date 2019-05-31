import numpy as np
import torch as t

def gelu(x):
    cdf = 0.5 * (1.0 + t.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * t.pow(x, 3))
    ))
    return x * cdf