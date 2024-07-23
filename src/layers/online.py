import torch
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn


class ConvolutionSignal(nn.Module):
    def __init__(self, d_input, d_output,
                 min_radius=0.0,
                 max_radius=1.0,
                 field='real'):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.field = field

        reservoir = ReservoirMatrix(d_in=self.d_input, d_out=self.d_output)
        kernel = reservoir.uniform_ring(min_radius=min_radius, max_radius=max_radius, field=self.field)
        self.K = nn.Parameter(kernel, requires_grad=True)

    def step(self, u):
        u = torch.einsum('ph, bh -> bp', self.K, u)

        return u

    def forward(self, u):
        u = torch.einsum('ph, bhl -> bpl', self.K, u)

        return u
