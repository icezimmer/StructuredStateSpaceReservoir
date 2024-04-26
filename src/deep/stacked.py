import torch.nn as nn
from src.reservoir.layers import LinearReservoir, LinearStructuredReservoir
from src.models.s4r.s4r import S4R
from src.models.esn.esn import ESN
from src.models.s4d.s4d import S4D
import torch


class StackedNetwork(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, d_output,
                 encoder, to_vec, decoder,
                 layer_dropout=0.0,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """
        encoder_models = ['conv1d', 'reservoir', 'structured_reservoir']
        decoder_models = ['conv1d', 'reservoir']

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(encoder_models))

        if decoder not in decoder_models:
            raise ValueError('Decoder must be one of {}'.format(decoder_models))

        super().__init__()

        if encoder == 'conv1d':
            self.encoder = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=1)
        elif encoder == 'reservoir':
            self.encoder = LinearReservoir(d_input=d_input, d_output=d_model, field='real')
        elif encoder == 'structured_reservoir':
            self.encoder = LinearStructuredReservoir(d_input=d_input, d_output=d_model, field='real')

        self.layers = nn.ModuleList([block_cls(d_model=d_model, **block_args) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                                       for _ in range(n_layers)])
        self.to_vec = to_vec

        if decoder == 'conv1d':
            self.decoder = nn.Conv1d(in_channels=d_model, out_channels=d_output, kernel_size=1)
        elif decoder == 'reservoir':
            self.decoder = LinearReservoir(d_input=d_model, d_output=d_output, field='real')

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, L)
        return:
            x: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        for layer, dropout in zip(self.layers, self.dropouts):
            x, _ = layer(x)
            x = dropout(x)

        if self.to_vec:
            x = self.decoder(x[:, :, -1:]).squeeze(-1)  # (B, d_model, L) -> (B, d_output)
        else:
            x = self.decoder(x)  # (*, d_model) -> (*, d_output)

        return x


class StackedReservoir(nn.Module):
    def __init__(self, n_layers, d_input, d_model,
                 encoder, transient,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """
        encoder_models = ['reservoir', 'structured_reservoir']

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(encoder_models))

        super().__init__()

        self.n_layers = n_layers

        if encoder == 'reservoir':
            self.encoder = LinearReservoir(d_input=d_input, d_output=d_model, field='real')
        elif encoder == 'structured_reservoir':
            self.encoder = LinearStructuredReservoir(d_input=d_input, d_output=d_model, field='real')

        self.layers = nn.ModuleList([S4R(d_model=d_model, **block_args) for _ in range(self.n_layers)])

        self.transient = transient

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        u = self.encoder.step(u)
        x_list = []
        for layer in self.layers:
            u, x = layer.step(u, x)
            x_list.append(x)
        x = torch.cat(tensors=x_list, dim=1)

        return u, x

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, L)
        return:
            x: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        x_list = []
        for layer in self.layers:
            x, _ = layer(x)  # (B, d_model, L) -> (B, d_model, L)
            x_list.append(x)

        x = torch.cat(tensors=x_list, dim=1)
        x = x[:, :, self.transient:]  # (B, d_model, L) -> (B, d_model, L - transient)

        return x


class StackedEchoState(nn.Module):
    def __init__(self, n_layers, d_input, d_model,
                 transient,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """

        super().__init__()

        self.n_layers = n_layers

        self.layers = nn.ModuleList([ESN(d_input=d_input, d_state=d_model, **block_args) for _ in range(self.n_layers)])

        self.transient = transient

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        x_list = []
        for layer in self.layers:
            x = layer.step(u, x)
            x_list.append(x)
        x = torch.cat(tensors=x_list, dim=1)

        return u, x

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, L)
        return:
            x: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        x_list = []
        for layer in self.layers:
            x, _ = layer(x)  # (B, d_model, L) -> (B, d_model, L)
            x_list.append(x)

        x = torch.cat(tensors=x_list, dim=1)
        x = x[:, :, self.transient:]  # (B, d_model, L) -> (B, d_model, L - transient)

        return x
