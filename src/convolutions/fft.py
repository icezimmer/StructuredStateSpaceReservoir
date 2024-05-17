import torch
import torch.nn as nn
from src.reservoir.vector import ReservoirVector
from src.kernels.vandermonde import (Vandermonde, VandermondeFreezeB, VandermondeFreezeC,
                                     VandermondeFreezeBC,
                                     VandermondeFreezeA,
                                     VandermondeFreezeAB, VandermondeFreezeAC)
from src.kernels.vandermonde_reservoir import VandermondeReservoir
from src.kernels.mini_vandermonde import (MiniVandermonde, MiniVandermondeFreezeW,
                                          MiniVandermondeFreezeA)
from src.kernels.mini_vandermonde_reservoir import MiniVandermondeReservoir


class FFTConv(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel,
                 min_scaleD=0.0, max_scaleD=1.0, dropout=0.0, drop_kernel=0.0, **kernel_args):
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
            'V': Vandermonde,
            'V-freezeB': VandermondeFreezeB,
            'V-freezeC': VandermondeFreezeC,
            'V-freezeBC': VandermondeFreezeBC,
            'V-freezeA': VandermondeFreezeA,
            'V-freezeAB': VandermondeFreezeAB,
            'V-freezeAC': VandermondeFreezeAC,
            'V-freezeABC': VandermondeReservoir,
            'miniV': MiniVandermonde,
            'miniV-freezeW': MiniVandermondeFreezeW,
            'miniV-freezeA': MiniVandermondeFreezeA,
            'miniV-freezeAW': MiniVandermondeReservoir,
        }

        if kernel not in kernel_classes.keys():
            raise ValueError('Kernel must be one of {}'.format(kernel_classes))

        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # SISO model

        self.kernel_cls = kernel_classes[kernel](d_input=self.d_input, d_state=self.d_state, **kernel_args)

        input2output_reservoir = ReservoirVector(d=self.d_input)
        D = input2output_reservoir.uniform_ring(min_radius=min_scaleD, max_radius=max_scaleD, field='real')
        self.D = nn.Parameter(D, requires_grad=True)  # (H,)

        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.activation = nn.Tanh()

    def _freeze_parameter(self, param_name):
        """
        Converts a parameter to a buffer, effectively freezing it.
        This means the parameter will no longer be updated during training.

        Args:
            param_name (str): The name of the parameter to freeze.
        """
        # Ensure the attribute exists and is a parameter
        if hasattr(self, param_name) and isinstance(getattr(self, param_name), nn.Parameter):
            # Convert to buffer
            param = getattr(self, param_name).data
            delattr(self, param_name)  # Remove as parameter
            self.register_buffer(param_name, param)  # Register as buffer
        else:
            raise ValueError(f"{param_name} is not a parameter in this module.")

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
        y = self.activation(y)

        return y, x

    def forward(self, u):
        """
        Apply the convolution to the input sequence (SISO model):
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        """
        k, _ = self.kernel_cls()
        k = self.drop_kernel(k)

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)
        k_s = torch.fft.fft(k, dim=-1)  # (H, L)

        y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, k_s), dim=-1)  # (B, H, L)
        y = y.real + torch.einsum('h,bhl->bhl', self.D, u)  # (B, H, L)

        y = self.drop(y)
        y = self.activation(y)

        return y, None


class FFTConvFreezeD(FFTConv):
    def __init__(self, d_input, d_state, kernel, min_scaleD=0.0, max_scaleD=1.0,
                 dropout=0.0, drop_kernel=0.0, **kernel_args):
        """
        Construct a discrete LTI SSM model whit frozen D.
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
        super().__init__(d_input, d_state, kernel, min_scaleD, max_scaleD, dropout, drop_kernel, **kernel_args)

        self._freeze_parameter('D')

    def forward(self, u):
        """
        Apply the convolution to the input sequence (SISO model):
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        """
        k, _ = self.kernel_cls()
        k = self.drop_kernel(k)

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)
        k_s = torch.fft.fft(k, dim=-1)  # (H, L)

        y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, k_s), dim=-1)  # (B, H, L)

        y = y.real + torch.einsum('h,bhl->bhl', self.D, u)  # (B, H, L)

        y = self.drop(y)
        y = self.activation(y)

        return y, None
