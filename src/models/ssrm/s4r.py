import torch
import torch.nn as nn
from src.models.conv.vandermonde import VandermondeConv, VandermondeReservoirConv


"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S4R(torch.nn.Module):
    def __init__(self, d_input, **layer_args):
        """
        Construct an SSM model with frozen state matrix Lambda_bar:
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        super().__init__()

        self.d_input = d_input
        self.d_output = d_input

        self.layer = VandermondeReservoirConv(d_input, **layer_args)

        self.mix_and_gate = nn.Sequential(
            nn.Conv1d(self.d_output, 2 * self.d_output, kernel_size=1),  # mix and double the number of features
            nn.GLU(dim=-2),  # gating
        )

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        y, x = self.layer.step(u, x)
        y = self.mix_and_gate(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the S4R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """

        y, _ = self.layer(u)  # (B, H, L)
        y = self.mix_and_gate(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
