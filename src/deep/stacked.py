import torch.nn as nn
from src.reservoir.layers import LinearReservoir, ZeroAugmentation, Truncation, LinearStructuredReservoir


class StackedNetwork(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, d_output,
                 encoder, to_vec, decoder,
                 layer_dropout=0.0,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """

        encoder_models = {
            'conv1d': nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=1),
            'reservoir': LinearReservoir(d_input=d_input, d_output=d_model,
                                         field='real'),
            'structured_reservoir': LinearStructuredReservoir(d_input=d_input, d_output=d_model,
                                                              field='real'),
            'pad': ZeroAugmentation(d_input=d_input, d_output=d_model)
        }

        decoder_models = {
            'conv1d': nn.Conv1d(in_channels=d_model, out_channels=d_output, kernel_size=1),
            'reservoir': LinearReservoir(d_input=d_model, d_output=d_output,
                                         field='real'),
            'structured_reservoir': LinearStructuredReservoir(d_input=d_model, d_output=d_output,
                                                              field='real'),
            'truncate': Truncation(d_input=d_model, d_output=d_output)
        }

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(list(encoder_models.keys())))
        if decoder not in decoder_models:
            raise ValueError('Decoder must be one of {}'.format(list(encoder_models.keys())))

        super().__init__()
        self.encoder = encoder_models[encoder]
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **block_args) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                                       for _ in range(n_layers)])
        self.to_vec = to_vec
        self.decoder = decoder_models[decoder]

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
