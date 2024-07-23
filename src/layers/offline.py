import torch
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, d_input, d_output, to_vec):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        self.to_vec = to_vec

        structured_reservoir = ReservoirMatrix(d_in=d_output, d_out=d_input)  # transpose of matrix (left multipl.)
        self.register_buffer('W_out_t',
                             structured_reservoir.uniform_ring(max_radius=1.0, min_radius=0.0, field='real'))  # (P+1,K)

    def _one_hot_encoding(self, labels):
        y = torch.nn.functional.one_hot(input=labels, num_classes=self.d_output)
        y = y.to(dtype=torch.float32)
        return y

    def forward(self, X, y=None):
        if y is not None:
            if self.to_vec:
                y = self._one_hot_encoding(y)

            self.W_out_t = torch.linalg.pinv(X).mm(y)  # (P, K)
            o = None
        else:
            o = torch.einsum('np,pk -> nk', X, self.W_out_t)  # prediction
            o = torch.argmax(o, dim=1) if self.to_vec else o

        return o


class RidgeRegression(LinearRegression):
    def __init__(self, d_input, d_output, to_vec, lambda_):
        if lambda_ <= 0:
            raise ValueError("Regularization lambda must be positive for Ridge Regression.")

        super().__init__(d_input, d_output, to_vec)

        self.lambda_ = lambda_

        self.register_buffer('eye_matrix', torch.eye(n=self.d_input, dtype=torch.float32))

    def forward(self, X, y=None):
        if y is not None:
            if self.to_vec:
                y = self._one_hot_encoding(y)

            self.W_out_t = torch.linalg.inv(X.t().mm(X) + self.lambda_ * self.eye_matrix).mm(X.t()).mm(y)  # (P, K)
            o = None
        else:
            o = torch.einsum('np,pk -> nk', X, self.W_out_t)  # prediction
            o = torch.argmax(o, dim=1) if self.to_vec else o

        return o