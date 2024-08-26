import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super(EmbeddingModel, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=self.padding_idx)

    def _compute_lengths(self, batch_data):
        """
        Compute the length of each padded time series in a batch.

        :param batch_data: Tensor of shape (B, L), where B is the batch size
                           and L is the sequence length.
        :return: Tensor of shape (B,), containing the lengths of each sequence.
        """
        # Create a mask where non-padding elements are True
        non_padding_mask = batch_data != self.padding_idx  # Shape: (B, L)

        # Sum along the time dimension to get the length of each sequence
        lengths = non_padding_mask.sum(dim=-1)  # Shape: (B,)

        return lengths

    def forward(self, x):
        lengths = self._compute_lengths(x)
        x = self.embedding(x)  # (B, L) -> (B, L, d_model)
        x = x.permute(0, 2, 1)  # (B, L, d_model) -> (B, d_model, L)
        return x, lengths
