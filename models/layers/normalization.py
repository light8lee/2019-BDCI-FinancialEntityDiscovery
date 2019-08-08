import torch

def normalize_adjs(input_masks, input_adjs):
    # this is equvalient to D^{-1/2} A D^{-1/2}
    row_sum = input_adjs.sum(-1) + 1e-7  # [b, t]
    d_inv_sqrt = torch.pow(row_sum, -0.5)  # [b, t]
    d_inv_sqrt *= input_masks  # [b, t], keep padding values to 0
    normalization = d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1) # [b, t, t]
    input_adjs = input_adjs * normalization
    return input_adjs