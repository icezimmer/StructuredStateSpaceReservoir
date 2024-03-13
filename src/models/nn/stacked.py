import torch
import torch.nn as nn


class NaiveStacked(nn.Module):
    def __init__(self, block, n_layers):
        super(NaiveStacked, self).__init__()
        self.d_output = block.d_output
        self.layers = nn.ModuleList([block for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x, None
