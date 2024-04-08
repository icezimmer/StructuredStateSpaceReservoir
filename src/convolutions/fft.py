import torch
from src.reservoir.state_reservoir import DiscreteStateReservoir, ContinuousStateReservoir
import torch.nn as nn


class FFTConv(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, kernel_cls, dropout=0.0, **kernel_args):
        """
        Construct an SSM model with frozen state matrix Lambda_bar:
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        # TODO: Delta trainable parameter not fixed to ones for continuous dynamics:
        #   Lambda_bar = Lambda_Bar(Lambda, Delta), B_bar = B(Lambda, B, Delta)

        super().__init__()

        self.d_input = d_model
        self.d_model = d_model
        self.d_output = d_model

        self.kernel = kernel_cls(d_model=self.d_model, **kernel_args)

        self.D = nn.Parameter(torch.randn(self.d_input, self.d_output, dtype=torch.float32),
                              requires_grad=True)  # (H, H)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.activation = nn.Tanh()

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        y, x = self.kernel.step(u, x)
        y = y + torch.einsum('hh,bh->bh', self.D, u)  # (B,H)
        y = self.activation(y)

        return y, x

    def forward(self, u):
        """
        Apply the convolution to the input sequence
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """

        k, _ = self.kernel()

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)
        k_s = torch.fft.fft(k, dim=-1)  # (H, L)

        y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, k_s), dim=-1)  # (B, H, L)
        y = y.real + torch.einsum('hh,bhl->bhl', self.D, u)  # (B, H, L)

        y = self.drop(y)
        y = self.activation(y)

        return y, None
