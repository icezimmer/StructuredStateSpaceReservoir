import torch.nn as nn
import torch


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        x = self.embedding(x)  # (B, L) -> (B, L, d_model)
        x = x.permute(0, 2, 1)  # (B, L, d_output) -> (B, d_output, L)
        return x