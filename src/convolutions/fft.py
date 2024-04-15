import torch
from src.reservoir.matrices import Reservoir
import torch.nn as nn


class FFTConv(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_cls, dropout=0.0, drop_kernel=0.0, **kernel_args):
        """
        Construct a discrete LTI SSM model.
        Recurrence view:
            x_new = A * x_old + B * u_new
            y_new = C * x_new + D * u_new
        Convolution view:
            y = conv_1d( u, kernel(A, B, C, length(u)) ) + D * u
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # SISO model

        self.kernel = kernel_cls(d_input=self.d_input, d_state=self.d_state, **kernel_args)

        input2output_reservoir = Reservoir(d_in=self.d_input, d_out=self.d_output)
        D = input2output_reservoir.uniform_disk_matrix(radius=1.0, field='real')
        self.D = nn.Parameter(D, requires_grad=True)  # (H, H)

        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.activation = nn.Tanh()

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation:
            x_new, y_new = kernel.step(u_new, x_old)
            y_new = y_new + D * u_new
        :param u: time step input of shape (B, H)
        :param x: time step state of shape (B, P)
        :return: y: time step output of shape (B, H), x: time step state of shape (B, P)
        """
        y, x = self.kernel.step(u, x)
        y = y + torch.einsum('hh,bh->bh', self.D, u)  # (B,H)
        y = self.activation(y)

        return y, x

    def forward(self, u):
        """
        Apply the convolution to the input sequence (SISO model):
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        """
        k, _ = self.kernel()
        k = self.drop_kernel(k)

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)
        k_s = torch.fft.fft(k, dim=-1)  # (H, L)

        y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, k_s), dim=-1)  # (B, H, L)
        y = y.real + torch.einsum('hh,bhl->bhl', self.D, u)  # (B, H, L)

        y = self.drop(y)
        y = self.activation(y)

        return y, None
