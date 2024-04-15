import torch
from src.reservoir.matrices import Reservoir
import torch.nn as nn


class LinearReservoir(nn.Module):
    def __init__(self, d_input, d_output,
                 radius=1.0,
                 field='real'):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.field = field

        reservoir = Reservoir(d_in=self.d_input, d_out=self.d_output)
        W_in = reservoir.uniform_disk_matrix(radius=radius, field=field)
        self.register_buffer('W_in', W_in)

    def forward(self, u):
        u = torch.einsum('ph, bhl -> bpl', self.W_in, u)

        return u


class NaiveEncoder(nn.Module):
    def __init__(self, d_input, d_output):
        if d_output < d_input:
            raise ValueError('d_output must be greater than d_input')
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output

    def forward(self, u):

        # concatenate the input with zeros
        u = torch.cat([u,
                       torch.zeros(u.shape[0],
                                   self.d_output - self.d_input,
                                   u.shape[2],
                                   dtype=u.dtype, device=u.device)],
                      dim=1)

        return u


class NaiveDecoder(nn.Module):
    def __init__(self, d_input, d_output):
        if d_output > d_input:
            raise ValueError('d_output be less than d_input')
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output

    def forward(self, u):
        # Take only the first d_output dimensions
        u = u[:, :self.d_output, :]

        if self.field == 'complex':
            u = u.real

        return u
