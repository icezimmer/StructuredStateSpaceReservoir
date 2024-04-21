import torch.nn as nn
from src.reservoir.layers import LinearReservoir, LinearStructuredReservoir
from src.models.s4r.s4r import S4R
import torch


class StackedReservoir(nn.Module):
    def __init__(self, n_layers, d_input, d_model,
                 encoder, transient, step,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """
        encoder_models = ['reservoir', 'structured_reservoir']

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(encoder_models))

        super().__init__()

        if encoder == 'reservoir':
            self.encoder = LinearReservoir(d_input=d_input, d_output=d_model, field='real')
        elif encoder == 'structured_reservoir':
            self.encoder = LinearStructuredReservoir(d_input=d_input, d_output=d_model, field='real')

        self.layers = nn.ModuleList([S4R(d_model=d_model, **block_args) for _ in range(n_layers)])

        self.transient = transient
        self.step = step

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, L)
        return:
            x: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        with torch.no_grad():
            x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

            for layer in self.layers:
                x, _ = layer(x)

            x = x[:, :, self.transient::self.step]  # (B, d_model, L) -> (B, d_output, 1)

        return x
