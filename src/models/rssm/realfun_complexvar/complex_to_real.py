import torch
import torch.nn as nn


class Real(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.real
        return x


class RealReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.real
        x = self.activation(x)
        return x


class RealTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.real
        x = self.activation(x)
        return x


class RealImagTanhGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
        self.glu = nn.GLU(dim=-2)

    def forward(self, x):
        x = torch.cat((x.real, x.imag), dim=-2)
        x = self.activation(x)
        x = self.glu(x)

        return x


class ABSTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = torch.abs(x)
        x = self.activation(x)

        return x


class AngleTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = torch.angle(x)
        x = self.activation(x)

        return x
