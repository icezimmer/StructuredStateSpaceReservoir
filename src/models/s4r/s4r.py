import torch
import torch.nn as nn
from src.reservoir.layers import LinearReservoir, LinearStructuredReservoir
from src.convolutions.fft import FFTConvReservoir

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


# TODO: replace mixing layer with only a non-linearity (identity is the best choice for mixing layer)
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

        kernel_classes = ['Vr', 'miniVr']

        if kernel not in kernel_classes:
            raise ValueError('Kernel must be one of {}'.format(kernel_classes))

        mixing_layers = ['identity', 'identity+tanh', 'reservoir+tanh']
        if mixing_layer not in mixing_layers:
            raise ValueError('Kernel must be one of {}'.format(mixing_layers))

        super().__init__()

        self.d_model = d_model

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model, kernel=kernel,
                                      **layer_args)

        if mixing_layer == 'identity':
            self.mix = nn.Identity()
            self.nl = nn.Identity()
        elif mixing_layer == 'identity+tanh':
            self.mix = nn.Identity()
            self.nl = nn.Tanh()
        elif mixing_layer == 'reservoir+tanh':
            self.mix = LinearReservoir(d_input=d_model, d_output=d_model, field='real')
            self.nl = nn.Tanh()

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
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
        y, _ = self.layer(u)
        y = self.mix(y)
        y = self.nl(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
