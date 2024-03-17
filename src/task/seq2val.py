import torch.nn as nn


class Seq2Val(nn.Module):
    """
    Wrapper class for SSM models. From a sequence of inputs, it produces a single output.
    """

    def __init__(self, model):
        super(Seq2Val, self).__init__()
        self.model = model  # Instance of the model

        if not hasattr(self.model, 'd_output'):
            raise AttributeError("The model must have a 'd_output' attribute specifying the output feature dimension.")

        # Classification layer to take the pooled output
        self.output_layer = nn.Linear(in_features=self.model.d_output, out_features=1)

    def forward(self, x):
        # Pass input through model
        y, _ = self.model(x)  # y shape is (B, H, L)

        y_pooled = y[:, :, -1]  # Now y_pooled is (B, H)

        # Process the output through the classification layer
        output = self.output_layer(y_pooled)  # output is (B, 1)
        return output
