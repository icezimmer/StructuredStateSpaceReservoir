import torch
import torch.nn as nn
from src.convolutions.fft import FFTConv

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S4R(torch.nn.Module):
    def __init__(self, d_model,
                 dropout=0.0,
                 **layer_args):
        """
        S4R model.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        super().__init__()

        self.d_model = d_model

        self.layer = FFTConv(d_input=self.d_model, d_state=self.d_model, **layer_args)

        self.mix_and_gate = nn.Sequential(
            nn.Conv1d(self.d_model, 2 * self.d_model, kernel_size=1),  # mix and double the num of features (no context)
            nn.GLU(dim=-2),  # gating
        )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        y, x = self.layer.step(u, x)
        y = self.mix_and_gate(y)
        y = self.drop(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the S4R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        y, _ = self.layer(u)
        y = self.mix_and_gate(y)
        y = self.drop(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
