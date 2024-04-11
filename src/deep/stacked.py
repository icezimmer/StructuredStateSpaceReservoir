import torch.nn as nn


class StackedNetwork(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, d_output,
                 layer_dropout=0.0, pre_norm=False, post_norm=False,
                 to_vec=False,
                 **block_args):
        super().__init__()
        self.to_vec = to_vec
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.encoder = nn.Conv1d(d_input, d_model, kernel_size=1)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **block_args) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                                       for _ in range(n_layers)])
        self.decoder = nn.Conv1d(d_model, d_output, kernel_size=1)

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, L)
        return:
            x: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            if self.pre_norm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            x, _ = layer(x)
            x = dropout(x)

            if self.post_norm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        if self.to_vec:
            x = self.decoder(x[:, :, -1:]).squeeze(-1)  # (B, d_model, L) -> (B, d_output)
        else:
            x = self.decoder(x)  # (*, d_model) -> (*, d_output)

        return x
