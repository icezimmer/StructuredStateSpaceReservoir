import torch
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn


class LinearReservoirRing(nn.Module):
    def __init__(self, d_input, d_output,
                 min_radius=0.0,
                 max_radius=1.0,
                 field='real',
                 length_last=True):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output

        self.field = field

        self.length_last = length_last

        reservoir = ReservoirMatrix(d_in=self.d_input, d_out=self.d_output)
        W_in = reservoir.uniform_ring(min_radius=min_radius, max_radius=max_radius, field=self.field)
        self.register_buffer('W_in', W_in)

    def step(self, u):
        u = torch.einsum('ph, bh -> bp', self.W_in, u)

        return u

    def forward(self, u):
        if self.length_last:
            u = torch.einsum('ph, bhl -> bpl', self.W_in, u)
        else:
            u = torch.einsum('ph, blh -> blp', self.W_in, u)

        return u


class LinearReservoirInterval(nn.Module):
    def __init__(self, d_input, d_output,
                 min_value=0.0,
                 max_value=1.0,
                 length_last=True):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output

        self.length_last = length_last

        reservoir = ReservoirMatrix(d_in=self.d_input, d_out=self.d_output)
        W_in = reservoir.uniform_interval(min_value=min_value, max_value=max_value)
        self.register_buffer('W_in', W_in)

    def step(self, u):
        u = torch.einsum('ph, bh -> bp', self.W_in, u)

        return u

    def forward(self, u):
        if self.length_last:
            u = torch.einsum('ph, bhl -> bpl', self.W_in, u)
        else:
            u = torch.einsum('ph, blh -> blp', self.W_in, u)

        return u
