import torch
import torch.nn as nn
from src.reservoir.vector import ReservoirVector
from src.models.rssm.kernels.vandermonde_reservoir import VandermondeReservoir


class FFTConvReservoir(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel,
                 min_scaleD=0.0, max_scaleD=1.0, **kernel_args):
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
        kernel_classes = {
            'Vr': VandermondeReservoir
        }

        if kernel not in kernel_classes.keys():
            raise ValueError('Kernel must be one of {}'.format(kernel_classes.keys()))

        super().__init__()

        self.d_input = d_input
        self.d_state = d_state  # d_state = P
        self.d_output = self.d_input  # SISO model (d_input = d_output = d_model = H)

        self.kernel_cls = kernel_classes[kernel](d_input=self.d_input, d_state=self.d_state, **kernel_args)
        K, _ = self.kernel_cls()
        self.register_buffer('K', K)  # (H, L)

        input2output_reservoir = ReservoirVector(d=self.d_input)
        D = input2output_reservoir.uniform_interval(min_value=min_scaleD, max_value=max_scaleD)  # (H,)
        self.register_buffer('D', D)  # (H,)

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model. Intended to be used during validation:
            x_new, y_new = kernel.step(u_new, x_old)
            y_new = y_new + D * u_new
        :param u: time step input of shape (B, H)
        :param x: time step state of shape (B, P)
        :return: y: time step output of shape (B, H), x: time step state of shape (B, P)
        """
        y, x = self.kernel_cls.step(u, x)
        y = y + torch.einsum('h,bh->bh', self.D, u)  # (B, H)

        return y, x

    def forward(self, u):
        """
        Apply the linear convolution to the input sequence (SISO model):
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        """
        # u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)
        # k_s = torch.fft.fft(self.K, dim=-1)  # (H, L)
        # y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, k_s), dim=-1)  # (B, H, L)
        # y = y + torch.einsum('h,bhl->bhl', self.D, u)  # (B, H, L)

        L = u.shape[-1]
        N = 2 * L - 1
        u_s = torch.fft.fft(u, n=N, dim=-1)  # (B, H, N)
        k_s = torch.fft.fft(self.K, n=N, dim=-1)  # (H, N)
        y = torch.fft.ifft(torch.einsum('bhs,hs->bhs', u_s, k_s), n=N, dim=-1)  # (B, H, N)
        y = y[..., :L]  # (B, H, L)
        y = y + torch.einsum('h,bhl->bhl', self.D, u)  # (B, H, L)

        return y, None
