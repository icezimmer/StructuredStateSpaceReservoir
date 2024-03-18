import torch.nn as nn


class Seq2Seq(nn.Module):
    """
    Wrapper class for SSM models. From a sequence of inputs, it produces a single output.
    """

    def __init__(self, model, d_vec):
        super(Seq2Seq, self).__init__()
        self.model = model  # Instance of the model
        self.d_vec = d_vec

        if not hasattr(self.model, 'd_output'):
            raise AttributeError("The model must have a 'd_output' attribute specifying the output feature dimension.")

        # Classification layer to take the pooled output
        self.output_layer = nn.Linear(in_features=self.model.d_output, out_features=self.d_vec)

    def forward(self, x):
        # Pass input through the model, expecting output shape of (B, H, L)
        y, _ = self.model(x)

        # Correct handling depends on the model's output. Assuming y shape is indeed (B, H, L):
        # Flatten (B, H, L) -> (B*L, H) for linear layer processing
        B, H, L = y.shape
        y_flattened = y.permute(0, 2, 1).reshape(B * L, H)  # Now (B*L, H)

        # Process the flattened output through the classification layer
        output = self.output_layer(y_flattened)  # Transforms to (B*L, d_vec)

        # Reshape back to sequence format (B, d_vec, L)
        output = output.reshape(B, L, self.d_vec).permute(0, 2, 1)  # Now (B, d_vec, L)

        return output
