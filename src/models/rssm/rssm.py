import torch
import torch.nn as nn
from src.reservoir.layers import LinearReservoir
from src.models.rssm.convolutions.fft_reservoir import FFTConvReservoir

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


# TODO: replace mixing layer with only a non-linearity (identity is the best choice for mixing layer)
class RSSM(torch.nn.Module):
    def __init__(self, d_model,
                 mixing_layer,
                 kernel,
                 **layer_args):
        """
        RSSM model.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """

        kernel_classes = ['Vr', 'miniVr']

        if kernel not in kernel_classes:
            raise ValueError('Kernel must be one of {}'.format(kernel_classes))

        mixing_layers = ['reservoir+tanh', 'reservoir+glu', 'identity']
        if mixing_layer not in mixing_layers:
            raise ValueError('Mixing Layer must be one of {}'.format(mixing_layers))

        super().__init__()

        self.d_model = d_model

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model, kernel=kernel,
                                      **layer_args)

        if mixing_layer == 'reservoir+tanh':
            self.mixing_layer = nn.Sequential(LinearReservoir(d_input=d_model, d_output=d_model, field='real'),
                                              nn.Tanh())
        elif mixing_layer == 'reservoir+glu':
            self.mixing_layer = nn.Sequential(LinearReservoir(d_input=d_model, d_output=2 * d_model, field='real'),
                                              nn.GLU(dim=-2))
        elif mixing_layer == 'identity':
            self.mixing_layer = nn.Identity()

    # TODO: implement step method for mixing layer
    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        u: (B, H)
        x: (B, P)
        Returns: y (B, H), state (B, P)
        """
        y, x = self.layer.step(u, x)
        # y = self.mixing_layer.step(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the RSSM model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        y, _ = self.layer(u)
        y = self.mixing_layer(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
