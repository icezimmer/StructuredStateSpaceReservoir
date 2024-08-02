import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_layers, d_input, d_output):

        super().__init__()

        # Construct mlp model, that is a multi layer perceptron made by mlp_layers of conv1d layers
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Conv1d(in_channels=d_input, out_channels=d_input, kernel_size=1))
            layers.append(nn.GLU(dim=-2))
            d_input = d_input // 2
            layers.append(nn.BatchNorm1d(d_input))

        layers.append(nn.Conv1d(in_channels=d_input, out_channels=d_output, kernel_size=1))

        self.mlp_layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        args:
            x: torch tensor of shape (B, d_input, 1)
        return:
            x: torch tensor of shape (B, d_output)
        """

        for layer in self.mlp_layers:  # (B, d_input, 1) - > (B, d_output, 1)
            x = layer(x)

        x = x.squeeze(-1)  # (B, d_output, 1) -> (B, d_output)

        return x
