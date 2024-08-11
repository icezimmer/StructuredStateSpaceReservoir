import torch
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn


class RidgeRegression(nn.Module):
    def __init__(self, d_input, d_output, alpha, to_vec, bias=True):
        if alpha < 0.0:
            raise ValueError("Regularization lambda must be positive for Ridge Regression.")
        
        super().__init__()

        self.bias = bias
        if self.bias:
            self.d_input = d_input + 1
        else:
            self.d_input = d_input
        self.d_output = d_output

        self.alpha = alpha

        self.to_vec = to_vec

        structured_reservoir = ReservoirMatrix(d_in=self.d_output, d_out=self.d_input)  # transpose of matrix (left multipl.)
        self.register_buffer('W_out_t',
                             structured_reservoir.uniform_ring(max_radius=1.0, min_radius=0.0, field='real'))  # (P+1,K)

        if self.alpha > 0.0:
            self.register_buffer('eye_matrix', torch.eye(n=self.d_input, dtype=torch.float32))

    def _one_hot_encoding(self, labels):
        y = torch.nn.functional.one_hot(input=labels, num_classes=self.d_output)
        y = y.to(dtype=torch.float32)
        return y

    def forward(self, X, y=None):
        if self.bias:
            X = torch.cat(tensors=(X, torch.ones(size=(X.shape[0], 1), device=X.device)), dim=-1)  # (N*(L-w), P+1)
        if y is not None:
            if self.to_vec:
                y = self._one_hot_encoding(y)

            if self.alpha == 0.0:
                self.W_out_t = torch.linalg.pinv(X).mm(y)  # (P, K)
            else:
                self.W_out_t = torch.linalg.inv(X.t().mm(X) + self.alpha * self.eye_matrix).mm(X.t()).mm(y)  # (P, K)
            o = None
        else:
            o = torch.einsum('np,pk -> nk', X, self.W_out_t)  # prediction
            o = torch.argmax(o, dim=1) if self.to_vec else o

        return o
