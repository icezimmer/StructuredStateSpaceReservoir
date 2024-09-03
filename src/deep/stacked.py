import torch.nn as nn
from src.layers.reservoir import LinearReservoirRing
from src.models.esn.esn import ESN
from src.layers.embedding import EmbeddingFixedPad, OneHotEncoding, Encoder
import torch


# TODO: Support transient for padded sequences with lengths
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
        encoder_models = ['conv1d', 'reservoir', 'embedding', 'onehot']
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
        elif encoder == 'embedding':
            self.encoder = EmbeddingFixedPad(vocab_size=d_input, d_model=d_model, padding_idx=0)
        elif encoder == 'onehot':
            self.encoder = OneHotEncoding(vocab_size=d_input, d_model=d_model,
                                          min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                          padding_idx=0)

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

    def forward(self, u, lengths=None):
        """
        args:
            u: torch tensor of shape (B, d_input, L)
            lengths: torch tensor of shape (B)
        return:
            y: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        y = self.encoder(u)  # (B, d_input, L) -> (B, d_model, L)

        for layer, dropout in zip(self.layers, self.dropouts):
            y, _ = layer(y)
            y = dropout(y)

        if self.to_vec:
            if lengths is not None:
                # Convert lengths to zero-based indices by subtracting 1
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2)  # Shape (B, 1, 1)

                # Expand indices to match the dimensions needed for gathering
                indices = indices.expand(y.shape[0], y.shape[1], 1)  # Shape (B, H, 1)
                y = y.gather(-1, indices)  # (B, d_model, L) -> (B, d_model, 1)
            else:
                y = y[:, :, -1:]  # (B, d_model, L) -> (B, d_model, 1)
            y = self.decoder(y).squeeze(-1)  # (B, d_model, L) -> (B, d_output)
        else:
            y = self.decoder(y)  # (B, d_model, L) -> (B, d_output, L)

        return y


class StackedReservoir(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, d_state, transient, take_last,
                 encoder, min_encoder_scaling=0.0, max_encoder_scaling=1.0,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """
        encoder_models = ['reservoir', 'onehot']

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(encoder_models))

        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        self.n_layers = n_layers

        self.take_last = take_last
        if self.take_last:
            self.d_output = self.d_model  # Take only the last layer output
        else:
            self.d_output = self.n_layers * self.d_model  # Take all the layers output concatenating them

        if encoder == 'reservoir':
            self.encoder = LinearReservoirRing(d_input=d_input, d_output=d_model,
                                               min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                               field='real')
        elif encoder == 'onehot':
            self.encoder = OneHotEncoding(vocab_size=d_input, d_model=d_model,
                                          min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                          padding_idx=0)

        self.layers = nn.ModuleList([block_cls(d_model=self.d_model, d_state=self.d_state, **block_args)
                                     for _ in range(self.n_layers)])

        self.transient = transient

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P, H)
        :return: output step (B, H), new state (B, P)
        """
        y = self.encoder.step(u)

        if self.take_last:
            x_list = []
            for i, layer in enumerate(self.layers):
                if x is not None:
                    x_i = x[:, i * self.d_state: (i + 1) * self.d_state, :]  # (B, P, H)
                else:
                    x_i = None
                y, z, x_i = layer.step(y, x_i)  # (B, H), (B, H), (B, P, H)
                x_list.append(x_i)
            x = torch.cat(tensors=x_list, dim=-2)  # (B, num_layers*P, H)
        else:
            x_list = []
            z_list = []
            for i, layer in enumerate(self.layers):
                if x is not None:
                    x_i = x[:, i * self.d_state: (i+1) * self.d_state, :]  # (B, P, H)
                else:
                    x_i = None
                y, z, x_i = layer.step(y, x_i)  # (B, H), (B, H), (B, P, H)
                x_list.append(x_i)
                z_list.append(z)
            x = torch.cat(tensors=x_list, dim=-2)  # (B, num_layers*P, H)
            z = torch.cat(tensors=z_list, dim=-1)  # (B, num_layers*H)

        return z, x

    def forward(self, u, lengths=None):
        """
        Forward method for the RSSM model.
        :param  u: input sequence, torch tensor of shape (B, d_input, L)
        :param  lengths: lengths of the input sequences, torch tensor of shape (B)
        :return: output sequence, torch tensor of shape (B, d_output, L - w)
        """
        y = self.encoder(u)  # (B, d_input, L) -> (B, d_model, L)

        if self.take_last:
            for layer in self.layers:
                y, z = layer(y)  # (B, d_model, L) -> (B, d_model, L)
            if lengths is not None:
                # Convert lengths to zero-based indices by subtracting 1
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2)  # Shape (B, 1, 1)

                # Expand indices to match the dimensions needed for gathering
                indices = indices.expand(z.shape[0], z.shape[1], 1)  # Shape (B, H, 1)
                z = z.gather(-1, indices)  # (B, d_model, L) -> (B, d_model, 1)
            else:
                z = z[:, :, self.transient:]
        else:
            z_list = []
            for layer in self.layers:
                y, z = layer(y)  # (B, d_model, L) -> (B, d_model, L)
                if lengths is not None:
                    # Convert lengths to zero-based indices by subtracting 1
                    indices = (lengths - 1).unsqueeze(1).unsqueeze(2)  # Shape (B, 1, 1)

                    # Expand indices to match the dimensions needed for gathering
                    indices = indices.expand(z.shape[0], z.shape[1], 1)  # Shape (B, H, 1)
                    z = z.gather(-1, indices)  # (B, d_model, L) -> (B, d_model, 1)
                else:
                    z = z[:, :, self.transient:]
                z_list.append(z)
            z = torch.cat(tensors=z_list, dim=-2)  # (B, num_layers * d_model, L - w)

        return z


class StackedEchoState(nn.Module):
    def __init__(self, n_layers, d_input, d_model,
                 transient, take_last, one_hot,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """

        super().__init__()

        self.d_input = d_input
        self.d_state = d_model

        self.n_layers = n_layers

        self.take_last = take_last
        if self.take_last:
            self.d_output = self.d_state  # Take only the last layer output
        else:
            self.d_output = self.n_layers * self.d_state  # Take all the layers output concatenating them

        self.layers = nn.ModuleList([ESN(d_input=self.d_input, d_state=self.d_state, **block_args)] +
                                    [ESN(d_input=self.d_state, d_state=self.d_state, **block_args)
                                    for _ in range(self.n_layers - 1)])

        self.one_hot = one_hot
        self.encoder = Encoder(w_in=self.layers[0].w_in, one_hot=self.one_hot)

        self.transient = transient

    def _one_hot_encoding(self, x):
        x = nn.functional.one_hot(input=x, num_classes=self.d_input)  # (*) -> (*, K=d_input)
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)  # (B, L, K) -> (B, K, L)
        x = x.to(dtype=torch.float32)
        return x

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: None, new state (B, P)
        """
        if self.one_hot:
            u = self._one_hot_encoding(u)
        if self.take_last:
            z_list = []
            for i, layer in enumerate(self.layers):
                if x is not None:
                    z = x[:, i * self.d_state: (i + 1) * self.d_state]
                else:
                    z = None
                z = layer.step(u, z)
                z_list.append(z)
            x = torch.cat(tensors=z_list, dim=-1)
            y = z
        else:
            z_list = []
            for i, layer in enumerate(self.layers):
                if x is not None:
                    z = x[:, i * self.d_state: (i+1) * self.d_state]
                else:
                    z = None
                z = layer.step(u, z)
                z_list.append(z)
            x = torch.cat(tensors=z_list, dim=-1)
            y = x

        return y, x

    def forward(self, x, lengths=None):
        """
        Forward method for the DeepESN model.
        :param  x: input sequence, torch tensor of shape (B, d_input, L)
        :param  lengths: lengths of the input sequences, torch tensor of shape (B)
        :return: output sequence, torch tensor of shape (B, d_state, L - w)
        """
        if self.one_hot:
            x = self._one_hot_encoding(x)
        if self.take_last:
            for layer in self.layers:
                x, _ = layer(x)  # (B, d_model, L) -> (B, d_model, L)
            if lengths is not None:
                # Convert lengths to zero-based indices by subtracting 1
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2)  # Shape (B, 1, 1)

                # Expand indices to match the dimensions needed for gathering
                indices = indices.expand(x.shape[0], x.shape[1], 1)  # Shape (B, H, 1)
                x = x.gather(-1, indices)  # (B, d_model, L) -> (B, d_model, 1)
            else:
                x = x[:, :, self.transient:]
        else:
            x_list = []
            for layer in self.layers:
                x, _ = layer(x)  # (B, d_model, L) -> (B, d_model, L)
                if lengths is not None:
                    # Convert lengths to zero-based indices by subtracting 1
                    indices = (lengths - 1).unsqueeze(1).unsqueeze(2)  # Shape (B, 1, 1)

                    # Expand indices to match the dimensions needed for gathering
                    indices = indices.expand(x.shape[0], x.shape[1], 1)  # Shape (B, H, 1)
                    x = x.gather(-1, indices)  # (B, d_model, L) -> (B, d_model, 1)
                else:
                    x = x[:, :, self.transient:]
                x_list.append(x)
            x = torch.cat(tensors=x_list, dim=-2)  # (B, num_layers * d_model, L - w)

        return x
