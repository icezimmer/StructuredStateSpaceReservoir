import torch.nn as nn
from src.layers.reservoir import LinearReservoirRing
from src.models.esn.esn import ESN
import torch


class StackedNetwork(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, d_output,
                 encoder, decoder, to_vec,
                 min_encoder_scaling=0.0, max_encoder_scaling=1.0,
                 min_decoder_scaling=0.0, max_decoder_scaling=1.0,
                 layer_dropout=0.0,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """
        encoder_models = ['conv1d', 'reservoir']
        decoder_models = ['conv1d', 'reservoir']

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(encoder_models))

        if decoder not in decoder_models:
            raise ValueError('Decoder must be one of {}'.format(decoder_models))

        super().__init__()

        if encoder == 'conv1d':
            self.encoder = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=1)
        elif encoder == 'reservoir':
            self.encoder = LinearReservoirRing(d_input=d_input, d_output=d_model,
                                               min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                               field='real')

        self.layers = nn.ModuleList([block_cls(d_model=d_model, **block_args) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                                       for _ in range(n_layers)])
        self.to_vec = to_vec

        if decoder == 'conv1d':
            self.decoder = nn.Conv1d(in_channels=d_model, out_channels=d_output, kernel_size=1)
        elif decoder == 'reservoir':
            self.decoder = LinearReservoirRing(d_input=d_model, d_output=d_output,
                                               min_radius=min_decoder_scaling, max_radius=max_decoder_scaling,
                                               field='real')

    def forward(self, u):
        """
        args:
            u: torch tensor of shape (B, d_input, L)
        return:
            y: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        y = self.encoder(u)  # (B, d_input, L) -> (B, d_model, L)

        for layer, dropout in zip(self.layers, self.dropouts):
            y, _ = layer(y)
            y = dropout(y)

        if self.to_vec:
            y = self.decoder(y[:, :, -1:]).squeeze(-1)  # (B, d_model, L) -> (B, d_output)
        else:
            y = self.decoder(y)  # (B, d_model, L) -> (B, d_output, L)

        return y


class StackedReservoir(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, transient,
                 min_encoder_scaling=0.0, max_encoder_scaling=1.0,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """

        super().__init__()

        self.d_state = d_model

        self.n_layers = n_layers

        self.d_output = self.n_layers * self.d_state

        self.encoder = LinearReservoirRing(d_input=d_input, d_output=self.d_state,
                                           min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                           field='real')

        self.layers = nn.ModuleList([block_cls(d_model=self.d_state, **block_args) for _ in range(self.n_layers)])

        self.transient = transient

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: output step (B, H), new state (B, P)
        """
        u = self.encoder.step(u)
        u_list = []
        x_list = []
        for layer in self.layers:
            u, x = layer.step(u, x)
            u_list.append(u)
            x_list.append(x)
        u = torch.cat(tensors=u_list, dim=-2)
        x = torch.cat(tensors=x_list, dim=-2)

        return u, x

    def forward(self, u):
        """
        Forward method for the RSSM model.
        :param  u: input sequence, torch tensor of shape (B, d_input, L)
        :return: output sequence, torch tensor of shape (B, d_output, L - w)
        """
        y = self.encoder(u)  # (B, d_input, L) -> (B, d_model, L)
        # y = torch.nn.ReLU()(y)

        x_list = []
        for layer in self.layers:
            y, x = layer(y)  # (B, d_model, L) -> (B, d_model, L)
            x_list.append(x[:, :, self.transient:])

        x = torch.cat(tensors=x_list, dim=-2)  # (B, num_layers * d_model, L - w)

        return x


class StackedEchoState(nn.Module):
    def __init__(self, n_layers, d_input, d_model,
                 transient,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """

        super().__init__()

        self.d_state = d_model

        self.n_layers = n_layers

        self.d_output = self.n_layers * self.d_state

        self.layers = nn.ModuleList([ESN(d_input=d_input, d_state=self.d_state, **block_args)] +
                                    [ESN(d_input=self.d_state, d_state=self.d_state, **block_args)
                                    for _ in range(self.n_layers - 1)])

        self.transient = transient

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: None, new state (B, P)
        """
        x_list = []
        for layer in self.layers:
            x = layer.step(u, x)
            x_list.append(x)
        x = torch.cat(tensors=x_list, dim=-2)

        return None, x

    def forward(self, x):
        """
        Forward method for the DeepESN model.
        :param  x: input sequence, torch tensor of shape (B, d_input, L)
        :return: output sequence, torch tensor of shape (B, d_state, L - w)
        """
        x_list = []
        for layer in self.layers:
            x, _ = layer(x)  # (B, d_model, L) -> (B, d_model, L)
            x_list.append(x[:, :, self.transient:])

        x = torch.cat(tensors=x_list, dim=-2)  # (B, num_layers * d_model, L - w)

        return x
