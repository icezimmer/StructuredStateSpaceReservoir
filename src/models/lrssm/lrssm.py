import torch
import torch.nn as nn
from src.models.rssm.convolutions.fft_reservoir import FFTConvReservoir
from src.models.lrssm.realfun_complexvar.complex_to_real import RealImag, Real

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class LRSSM(torch.nn.Module):
    def __init__(self, d_model,
                 kernel,
                 mlp_layers,
                 act,
                 dropout=0.0,
                 **reservoir_layer_args):
        """
        S4D model.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        kernel_classes = ['Vr']
        if kernel not in kernel_classes:
            raise ValueError('Kernel must be one of {}'.format(kernel_classes))

        activations = ['relu', 'tanh', 'glu']
        if act not in activations:
            raise ValueError('Activation function must be one of {}'.format(activations))

        super().__init__()

        self.d_model = d_model

        self.reservoir_layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model,
                                                kernel=kernel, discrete=False, **reservoir_layer_args)

        self.to_real = Real()

        if act == 'glu':
            act_f = nn.GLU(dim=-2)
        elif act == 'tanh':
            act_f = nn.Tanh()
        elif act == 'relu':
            act_f = nn.ReLU()
        else:
            raise ValueError("Activation function not recognized")

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        mlp = []
        for i in range(mlp_layers):
            if act == 'glu':
                mlp.append(nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=1))
            else:
                mlp.append(nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1))
            mlp.append(act_f)

        self.mlp = nn.Sequential(*mlp)

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        y, x = self.reservoir_layer.step(u, x)
        y = self.to_real(y)
        y = self.drop(y)
        y = self.mlp(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the S4D model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        y, _ = self.reservoir_layer(u)
        y = self.to_real(y)
        y = self.drop(y)
        y = self.mlp(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
