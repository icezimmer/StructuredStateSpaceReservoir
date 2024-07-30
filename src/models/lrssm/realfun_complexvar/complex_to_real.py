import torch
import torch.nn as nn


class Real(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.real
        return x


class RealImag(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat((x.real, x.imag), dim=-2)
        return x
