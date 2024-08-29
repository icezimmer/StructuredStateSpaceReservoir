import torch.nn as nn
import torch
from src.layers.reservoir import LinearReservoirRing


class EmbeddingFixedPad(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super(EmbeddingFixedPad, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=self.padding_idx)

    def forward(self, x):
        x = self.embedding(x)  # (B, L) -> (B, L, d_model)
        x = x.permute(0, 2, 1)  # (B, L, d_model) -> (B, d_model, L)
        return x


class OneHotEncoding(nn.Module):
    def __init__(self, vocab_size, d_model, min_radius, max_radius, padding_idx):
        super(OneHotEncoding, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.encoder = LinearReservoirRing(d_input=vocab_size, d_output=d_model,
                                           min_radius=min_radius, max_radius=max_radius,
                                           field='real',
                                           length_last=False)

    def _one_hot_encoding(self, x):
        x = nn.functional.one_hot(input=x, num_classes=self.vocab_size)  # (B, L) -> (B, L, K=vocab_size)
        x = x.to(dtype=torch.float32)
        return x

    def step(self, u):
        x = self._one_hot_encoding(u)  # (B) -> (B, vocab_size)
        x = self.encoder.step(x)  # (B, vocab_size) -> (B, d_model)
        x[u == self.padding_idx] = 0.0
        return x

    def forward(self, u):
        x = self._one_hot_encoding(u)  # (B, L) -> (B, L, vocab_size)
        x = self.encoder(x)  # (B, L, vocab_size) -> (B, L, d_model)
        x[u == self.padding_idx] = 0.0
        x = x.permute(0, 2, 1)   # (B, L, d_model) -> (B, dmodel, L)
        return x
