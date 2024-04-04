import torch.nn as nn


class ResidualNetwork(nn.Module):

    def __init__(self, block_factory, n_layers, d_input, d_model, d_output,
                 layer_dropout=0.0, pre_norm=False,
                 to_vec=False,
                 **block_args):
        super().__init__()
        self.to_vec = to_vec
        self.pre_norm = pre_norm

        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([block_factory(d_model=d_model, **block_args) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                                       for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, L)
        return:
            x: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        x = self.encoder(x.transpose(-1, -2)).transpose(-1, -2)  # (B, d_input, L) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):

            z = x
            if self.pre_norm:
                # Pre normalization
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)

            # Residual connection
            x = z + x  # (B, d_model, L) + (B, d_model, L) -> (B, d_model, L)

            if not self.pre_norm:
                # Post normalization
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        if self.to_vec:
            x = x[:, :, -1]  # (B, d_model, L) -> (B, d_model)
        else:
            x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L , d_model)

        x = self.decoder(x)  # (*, d_model) -> (*, d_output)

        return x
