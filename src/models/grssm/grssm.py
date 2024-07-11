import torch
import torch.nn as nn
from src.reservoir.layers import LinearReservoir
from src.models.grssm.convolutions.fft_reservoir import FFTConvReservoir

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


# TODO: replace mixing layer with only a non-linearity (identity is the best choice for mixing layer)
class gRSSM(torch.nn.Module):
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

        mixing_layers = ['reservoir+glu', 'glu']
        if mixing_layer not in mixing_layers:
            raise ValueError('Mixing Layer must be one of {}'.format(mixing_layers))

        super().__init__()

        self.d_model = d_model

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model, kernel=kernel,
                                      **layer_args)

        if mixing_layer == 'reservoir+glu':
            self.mixing_layer = nn.Sequential(LinearReservoir(d_input=2 * d_model, d_output=2 * d_model, field='real'),
                                              nn.GLU(dim=-2))
        elif mixing_layer == 'glu':
            self.mixing_layer = nn.GLU(dim=-2)

    # TODO: implement step method for mixing layer
    def step(self, u, x):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: output step (B, H), new state (B, P)
        """
        y, x = self.layer.step(u, x)
        # y = self.mixing_layer.step(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the RSSM layer.
        :param u: batched input sequence of shape (B, H, L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B, H, L) = (batch_size, d_output, input_length)
        """
        y, _ = self.layer(u)
        y = self.mixing_layer(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
