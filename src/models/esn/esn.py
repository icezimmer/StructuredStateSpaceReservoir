import torch
import torch.nn as nn
from src.reservoir.state import DiscreteStateReservoir
from src.reservoir.matrices import ReservoirMatrix


class ESN(nn.Module):
    """Class of Echo State Network model using PyTorch."""

    def __init__(self, d_input, d_state, input_scaling, bias_scaling, spectral_radius, leakage_rate):
        """
        Constructor of ESN model.
        """
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.leakage_rate = leakage_rate

        input2state_reservoir = ReservoirMatrix(d_in=self.d_input, d_out=self.d_state)
        w_in = input2state_reservoir.uniform_ring(max_radius=input_scaling, min_radius=0.0, field='real')
        self.register_buffer('w_in', w_in)

        self.register_buffer('x0', torch.zeros(self.d_state, dtype=torch.float32))

        state_reservoir = DiscreteStateReservoir(self.d_state)
        w_hh = state_reservoir.echo_state_matrix(max_radius=spectral_radius, leakage_rate=leakage_rate)
        self.register_buffer('w_hh', w_hh)

        self.register_buffer('bias',  2 * bias_scaling * torch.rand(self.d_state, dtype=torch.float32) - bias_scaling)
        self.nl = nn.Tanh()

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: new state (B, P)
        """
        if x is None:
            x = self.x0.unsqueeze(0).expand(u.size(0), -1)
        preactivation = (torch.einsum('ph, bh -> bp', self.w_in, u) +
                         torch.einsum('pp, bp -> bp', self.w_hh, x) +
                         self.bias)
        x = (1 - self.leakage_rate) * x + self.leakage_rate * self.nl(preactivation)
        return x

    def forward(self, u):
        """
        Forward method for the ESN model
        :param u: batched input sequence of shape (B, H, L) = (batch_size, d_input, input_length)
        :return: x: batched state sequence of shape (B, H, L) = (batch_size, d_state, input_length)
        """
        x_prev = self.x0.unsqueeze(0).expand(u.shape[0], -1)
        x_list = []
        for k in range(u.shape[-1]):
            x_curr = self.step(u[:, :, k], x_prev)
            x_list.append(x_curr.unsqueeze(-1))
            x_prev = x_curr
        return torch.cat(x_list, dim=-1), None
