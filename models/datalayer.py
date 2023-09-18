import torch
import torch.nn as nn

import numpy as np

from utils.fft import *

class DCLayer(nn.Module):
    """
        Data Consistency layer from DC-CNN, apply for single coil mainly
    """
    def __init__(
        self, 
        lambda_init=np.log(np.exp(1)-1.)/1., 
        learnable=True
        ):
        """
        Args:
            lambda_init (float): Init value of data consistency block (DCB)
        """
        super(DCLayer, self).__init__()
        self.learnable = learnable
        self.lambda_ = nn.Parameter(torch.ones(1) * lambda_init, requires_grad=self.learnable)

    def forward(self, x, y, mask):
        A_x = fftnc(x)
        k_dc = (1 - mask) * A_x + mask * (self.lambda_.to(x.device) * A_x + (1 - self.lambda_.to(x.device)) * y)
        x_dc = ifftnc(k_dc)
        return x_dc

    def extra_repr(self):
        return f"lambda={self.lambda_.item():.4g}, learnable={self.learnable}"