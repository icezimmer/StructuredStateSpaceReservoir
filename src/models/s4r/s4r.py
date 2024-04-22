import torch
import torch.nn as nn
from src.reservoir.layers import LinearReservoir, LinearStructuredReservoir
from src.convolutions.fft import FFTConvReservoir
from src.kernels.vandermonde import VandermondeReservoir
from src.kernels.mini_vandermonde import MiniVandermondeFullReservoir

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S4R(torch.nn.Module):
    def __init__(self, d_model,
                 mixing_layer,
                 kernel,
                 **layer_args):
        """
        S4R model.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """

        kernel_classes = ['V-freezeABC', 'miniV-freezeAW']

        if kernel not in kernel_classes:
            raise ValueError('Kernel must be one of {}'.format(kernel_classes))

        mixing_layers = ['reservoir+tanh', 'reservoir+glu', 'structured_reservoir+glu']
        if mixing_layer not in mixing_layers:
            raise ValueError('Kernel must be one of {}'.format(mixing_layers))

        super().__init__()

        self.d_model = d_model

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model, kernel=kernel,
                                      **layer_args)

        if mixing_layer == 'reservoir+tanh':
            self.mix = LinearReservoir(d_input=d_model, d_output=d_model, field='real')
            self.nl = nn.Tanh()
        elif mixing_layer == 'reservoir+glu':
            self.mix = LinearReservoir(d_input=d_model, d_output=2 * d_model, field='real')
            self.nl = nn.GLU(dim=-2)
        elif mixing_layer == 'structured_reservoir+glu':
            self.mixing_layer = LinearStructuredReservoir(d_input=d_model, d_output=2 * d_model, field='real')
            self.nl = nn.GLU(dim=-2)

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        with torch.no_grad():
            y, x = self.layer.step(u, x)
            y = self.mix.step(y)
            y = self.nl(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the S4R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        with torch.no_grad():
            y, _ = self.layer(u)
            y = self.mix(y)
            y = self.nl(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None