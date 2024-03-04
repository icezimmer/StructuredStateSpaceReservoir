import torch
import torch.nn as nn


class Seq2Val(nn.Module):
    def __init__(self, model):
        super(Seq2Val, self).__init__()
        self.model = model  # Instance of the model

        # Aggregate the output (B, H, L) across the L dimension.
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Modify the classification layer to take the pooled output
        # Since the pooled output will be (B, H, 1), we adjust the input features of the linear layer accordingly
        self.output_layer = nn.Linear(self.model.d_output, 1)

    def forward(self, x):
        # Pass input through model
        y, _ = self.model(x)  # y shape is (B, H, L)

        # Global average pooling across the length sequence L
        # y_pooled = self.global_avg_pool(y)
        y_pooled = y[:, :, -1]  # Now y_last is (B, H, 1) with L pooled to 1

        # Squeeze the last dimension to get (B, H)
        y_squeezed = torch.squeeze(y_pooled, -1)  # Now y_squeezed is (B, H)

        # Process the output through the classification layer
        output = self.output_layer(y_squeezed)  # output is (B, 1)

        return output
