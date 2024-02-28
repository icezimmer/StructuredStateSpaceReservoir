import torch
import torch.nn as nn


class S4DSeq2Val(nn.Module):
    def __init__(self, s4d_model):
        super(S4DSeq2Val, self).__init__()
        self.s4d_model = s4d_model  # Instance of the S4D model

        # Assuming the output of S4D model is (B, H, L)
        # You might want to aggregate this information across the L dimension.

        # Define a global pooling operation or another method of aggregation
        # For simplicity, let's use global average pooling here
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Modify the classification layer to take the pooled output
        # Since the pooled output will be (B, H, 1), we adjust the input features of the linear layer accordingly
        self.output_layer = nn.Linear(self.s4d_model.d_output, 1)

    def forward(self, x):
        # Pass input through S4D model
        y, _ = self.s4d_model(x)  # y shape is (B, H, L)

        # Global average pooling across the length sequence L
        y_pooled = self.global_avg_pool(y)  # Now y_pooled is (B, H, 1) with L pooled to 1

        # Squeeze the last dimension to get (B, H)
        y_squeezed = torch.squeeze(y_pooled, -1)  # Now y_squeezed is (B, H)

        # Process the output through the classification layer
        output = self.output_layer(y_squeezed)  # output is (B, 1)

        return output

# Example usage
# s4d_model = S4D(...)
# classifier = S4DClassifier(s4d_model)
# x = torch.randn(batch_size, input_channels, sequence_length)
# probs = classifier(x)  # probs shape will be (B, 1)
