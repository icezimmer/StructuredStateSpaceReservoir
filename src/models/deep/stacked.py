import torch.nn as nn


class NaiveStacked(nn.Module):
    def __init__(self, block_factory, n_layers, d_input, d_model, d_output, to_vec=False, *args, **kwargs):
        super().__init__()
        self.to_vec = to_vec
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([block_factory(d_model=d_model, *args, **kwargs) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = x.transpose(-1, -2)  # (B, d_input, L) -> (B, L, d_input)
        x = self.encoder(x)  # (*, d_input) -> (*, d_model)
        x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)

        for layer in self.layers:
            x, _ = layer(x)

        x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)
        if self.to_vec:
            x = x[:, -1, :]  # (B, L, d_model) -> (B, d_model)
        x = self.decoder(x)  # (*, d_model) -> (*, d_output)

        return x
