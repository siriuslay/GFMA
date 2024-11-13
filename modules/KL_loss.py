import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):  # output / out
        epsilon = 1e-10
        p_s = F.log_softmax(y_s / self.T, dim=1) + epsilon
        p_t = F.softmax(y_t / self.T, dim=1) + epsilon
        loss = F.kl_div(p_s, p_t, size_average=False, reduction='batchmean') * (self.T ** 2) / y_s.shape[0]
        return loss
