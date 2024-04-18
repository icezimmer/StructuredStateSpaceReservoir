import torch
import torch.nn as nn
from src.reservoir.layers import LinearReservoir, LinearStructuredReservoir

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S4R(torch.nn.Module):
    def __init__(self, d_model,
                 mixing_layer,
                 convolution_cls,
                 dropout=0.0,
                 **layer_args):
        """
        S4R model.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """

        mixing_layers = ['conv1d', 'conv1d+glu', 'reservoir', 'reservoir+glu', 'structured_reservoir',
                         'structured_reservoir+glu']

        if mixing_layer not in mixing_layers:
            raise ValueError('Mixing Layer must be one of {}'.format(mixing_layers))

        super().__init__()

        self.d_model = d_model

        self.layer = convolution_cls(d_input=self.d_model, d_state=self.d_model, **layer_args)

        if mixing_layer == 'conv1d':
            self.mixing_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)

        elif mixing_layer == 'conv1d+glu':
            self.mixing_layer = nn.Sequential(  # mix and double the num of features + gating
                                                nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=1),
                                                nn.GLU(dim=-2))
        elif mixing_layer == 'reservoir':
            self.mixing_layer = LinearReservoir(d_input=d_model, d_output=d_model, field='real')
        elif mixing_layer == 'reservoir+glu':
            self.mixing_layer = nn.Sequential(
                LinearReservoir(d_input=d_model, d_output=2 * d_model, field='real'),
                nn.GLU(dim=-2))
        elif mixing_layer == 'structured_reservoir':
            self.mixing_layer = LinearStructuredReservoir(d_input=d_model, d_output=d_model, field='real')
        elif mixing_layer == 'structured_reservoir+glu':
            self.mixing_layer = nn.Sequential(
                LinearStructuredReservoir(d_input=d_model, d_output=2 * d_model, field='real'),
                nn.GLU(dim=-2))

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        y, x = self.layer.step(u, x)
        y = self.mixing_layer(y)
        y = self.drop(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the S4R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        y, _ = self.layer(u)
        y = self.mixing_layer(y)
        y = self.drop(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
