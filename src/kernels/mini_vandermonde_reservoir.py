import torch
from src.reservoir.state import DiscreteStateReservoir
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn


class MiniVandermondeReservoir(nn.Module):
    """
    Generate convolution kernel from diagonal SSM parameters
    """

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability,
                 input_output_scaling=1.0,
                 field='complex'):
        """
        Construct the convolution kernel.
        Assuming diagonal state matrix A of shape (d_state), the Vandermonde Kernel is:
            kernel[i,l] = C[i,:] * diag(A)^l * B[:,i] = (C[i,:] .* B[:,i]) * A^l
        where:
            diag(A)[j,j] = A[j]
            A^l[j] = A[j]^l
        and:
            j = 0, ..., d_state-1
            i = 0, ..., d_input-1
            l = 0, ..., kernel_size-1.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # Necessary condition for the Vandermonde kernel (SISO)

        self.register_buffer('x0', torch.zeros(self.d_state, dtype=torch.complex64))

        input_output_reservoir = ReservoirMatrix(d_in=self.d_state, d_out=self.d_output)

        W = input_output_reservoir.uniform_disk(radius=input_output_scaling, field=field)

        state_reservoir = DiscreteStateReservoir(self.d_state)
        Lambda_bar = state_reservoir.diagonal_state_space_matrix(
            min_radius=strong_stability, max_radius=weak_stability, field=field)

        self.register_buffer('A', Lambda_bar)  # (P)
        self.register_buffer('W', W)  # (H, P)
        self.register_buffer('B', torch.sqrt(torch.transpose(self.W, 0, 1)))  # (P, H)
        self.register_buffer('C', torch.sqrt(self.W))  # (H, P)

        powers = torch.arange(kernel_size, dtype=torch.float32)
        V = self.A.unsqueeze(-1) ** powers  # (P, L)

        kernel = torch.einsum('hp,pl->hl', W, V)
        self.register_buffer('K', kernel)  # (H, L)

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
            x_new = A * x_old + u_new
            y_new = W * x_new,
        that is equivalent for y to:
            x_new = A * x_old + sqrt(W^t) * u_new
            y_new = sqrt(W) * x_new,
        :param u: time step input of shape (B, H)
        :param x: time step state of shape (B, P)
        :return: y: time step output of shape (B, H), x: time step state of shape (B, P)
        """
        u = u.to(dtype=torch.complex64)
        if x is None:
            x = self.x0.unsqueeze(0).expand(u.shape[0], -1)
        x = torch.einsum('p,bp->bp', self.A, x) + torch.einsum('ph,bh->bp', self.B, u)  # (B,P)
        y = torch.einsum('hp,bp->bh', self.C, x).real  # (B,H)

        return y, x

    def forward(self):
        """
        Generate the convolution kernel from the diagonal SSM parameters
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        kernel = self.K  # (H, L)

        return kernel, None
