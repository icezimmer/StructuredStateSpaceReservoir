import torch
import torch.nn as nn


class Seq2Val(nn.Module):
    """
    Wrapper class for SSM models. From a sequence of inputs, it produces a single output.
    """

    def __init__(self, model):
        super(Seq2Val, self).__init__()
        self.model = model  # Instance of the model

        # Aggregate the output (B, H, L) across the L dimension.
        #self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Modify the classification layer to take the pooled output
        # Since the pooled output will be (B, H, 1), we adjust the input features of the linear layer accordingly
        self.output_layer = nn.Linear(self.model.d_output, 1)

    def forward(self, x):
        # Pass input through model
        y, _ = self.model(x)  # y shape is (B, H, L)

        y_pooled = y[:, :, -1]  # Now y_pooled is (B, H)

        # Process the output through the classification layer
        output = self.output_layer(y_pooled)  # output is (B, 1)
        return output
