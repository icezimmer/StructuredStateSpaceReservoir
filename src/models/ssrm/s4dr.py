import math
import torch
import torch.nn as nn

from einops import repeat
from src.reservoir.state_reservoir import ContinuousStateReservoir
from src.models.nn.dropout import DropoutNd


"""
see: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
"""
class S4DRKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state=64, high_stability=-0.9, low_stability=-0.8, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()

        self.d_state = d_state

        # Generate dt
        log_dt = torch.rand(d_input, dtype=torch.float32) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)  # (H)

        d_output = d_input
        C = torch.randn(d_output, self.d_state, dtype=torch.complex64)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (H, N, 2)
        self.register("log_dt", log_dt, lr)

        # Generate complex diagonal matrix A
        if high_stability > low_stability or low_stability > 0:
            raise ValueError("For the continuous dynamics stability we must have: "
                             "'high_stability' < Re(lambda) <= 'low_stability' <= 0.")
        else:
            cr = ContinuousStateReservoir(self.d_state, high_stability, low_stability, 'complex')
            A = cr.diagonal_state_matrix()
            self.A = nn.Parameter(torch.view_as_real(A), requires_grad=False)  # (N, 2)

    def forward(self, input_length):
        """
        returns: (..., c, L) where c is number of channels (default 1) and L is the length of the input sequence
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H, N)
        A = torch.view_as_complex(self.A)  # (N)

        # repeat dt N=d_state times on dim = 1
        dt_n = repeat(dt, 'h-> h n', n=self.d_state)  # (H, N)

        # Vandermonde multiplication (see Kernel of DSS_EXP in the paper DSS):
        # On modern parallelizable hardware such as GPUs, a simple fast algorithm is to compute
        # Vandermonde multiplication with naive summation (using O(N L) operations),
        # but without materializing the Vandermonde matrix (using O(N + L) space) (see S4 with SSMKernelDiag
        # or S4DR with S4DRKernel).
        dtA = torch.einsum('n,hn->hn', A, dt_n)  # (H, N)
        K = dtA.unsqueeze(-1) * torch.arange(input_length, device=A.device)  # (H, N, L)
        C = C * (torch.exp(dtA) - 1.) / A  # (H, N)
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real  # (H, L)

        return K

    def register(self, name, tensor, lr=None):
        """
        Register a tensor as a parameter or buffer, with an associated learning rate.
        :param name:
        :param tensor:
        :param lr:
        :return:
        """

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4DR(nn.Module):
    def __init__(self, d_input, d_state=64, dropout=0.0, **kernel_args):
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input

        self.D = nn.Parameter(torch.randn(self.d_input, dtype=torch.float32))  # (H)

        # SSM Kernel
        self.kernel = S4DRKernel(self.d_input, d_state=self.d_state, **kernel_args)

        # Point-wise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.d_input, 2 * self.d_input, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        input_length = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(input_length=input_length)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*input_length)  # (H L)
        u_f = torch.fft.rfft(u, n=2*input_length)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2*input_length)[..., :input_length]  # (B H L)

        # Add Du term - essentially a skip connection
        y = y + torch.einsum('h, bhl -> bhl', self.D, u)  # (B H L)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified
