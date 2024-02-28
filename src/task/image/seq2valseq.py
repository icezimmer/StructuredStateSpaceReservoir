import torch
import torch.nn as nn


# Wrapper class for S4D
class Seq2ValSeq(nn.Module):
    def __init__(self, model):
        super(Seq2ValSeq, self).__init__()
        self.model = model  # Instance of the S4D model

        # Assuming the output of S4D model is (B, H, L), where B is batch size,
        # H is hidden dimension, and L is sequence length

        # Add a layer to process the output of S4D model.
        # This example uses a linear layer followed by a sigmoid activation function.
        # The linear layer reduces the dimension from H to 1, making it suitable for binary classification.
        self.output_layer = nn.Linear(self.model.d_output, 1)

    def forward(self, x):
        # Pass input through S4D model
        y, _ = self.model(x)

        # Process the output through the classification layer.
        # Assuming y shape is (B, H, L), we want to keep the batch and sequence dimensions
        # and squeeze the hidden dimension to get (B, L) output after sigmoid.
        # This means we apply the linear layer across each sequence element.
        output = self.classification_layer(y.transpose(1, 2))  # Transpose to (B, L, H) for Linear layer
        output = torch.squeeze(output, -1)  # Squeeze to remove the last dimension, resulting in (B, L)

        return output
