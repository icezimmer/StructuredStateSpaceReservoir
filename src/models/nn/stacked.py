import torch.nn as nn


class NaiveStacked(nn.Module):
    def __init__(self, block_factory, n_layers, *args, **kwargs):
        super(NaiveStacked, self).__init__()
        self.layers = nn.ModuleList([block_factory(*args, **kwargs) for _ in range(n_layers)])

        # Assuming the last block's output dimension is representative for the whole stack
        if n_layers > 0:
            self.d_output = self.layers[-1].d_output
        else:
            self.d_output = None

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x, None
