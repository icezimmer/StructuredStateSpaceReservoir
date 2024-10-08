import torch
from src.reservoir.state import DiscreteStateReservoir, ContinuousStateReservoir
from src.reservoir.matrices import ReservoirMatrix
from src.reservoir.vector import ReservoirVector
import torch.nn as nn


class VandermondeReservoir(nn.Module):
    """
    Generate convolution kernel from diagonal SSM parameters
    """

    def __init__(self, d_input, d_state, kernel_size,
                 discrete,
                 strong_stability, weak_stability,
                 low_oscillation, high_oscillation,
                 min_scaleB=0.0, max_scaleB=1.0,
                 min_scaleC=0.0, max_scaleC=1.0,
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
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        if not discrete:
            if low_oscillation <= 0.0 or high_oscillation <= 0.0:
                raise ValueError("For Continuous SSM delta time must be positive.")
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # Necessary condition for the Vandermonde kernel (SISO)

        self.register_buffer('x0', torch.zeros(self.d_state, self.d_input, dtype=torch.complex64))

        input2state_reservoir = ReservoirMatrix(d_in=self.d_input, d_out=self.d_state)
        state2output_reservoir = ReservoirMatrix(d_in=self.d_state, d_out=self.d_output)

        B = input2state_reservoir.uniform_ring(min_radius=min_scaleB, max_radius=max_scaleB, field='complex')  # (P, H)
        C = state2output_reservoir.uniform_ring(min_radius=min_scaleC, max_radius=max_scaleC, field='complex')  # (H, P)

        if discrete:
            state_reservoir = DiscreteStateReservoir(self.d_state)
            Lambda_bar = state_reservoir.diagonal_state_space_matrix(
                         min_radius=strong_stability, max_radius=weak_stability,
                         min_theta=low_oscillation, max_theta=high_oscillation,
                         field=field, n_ssm=self.d_input)
            B_bar = self._normalization(Lambda_bar, B)
        else:
            state_reservoir = ContinuousStateReservoir(self.d_state)
            Lambda = state_reservoir.diagonal_state_space_matrix(
                     min_real_part=strong_stability, max_real_part=weak_stability, field=field, n_ssm=self.d_input)
            rate_reservoir = ReservoirVector(d=self.d_input)
            dt = rate_reservoir.uniform_interval(min_value=low_oscillation, max_value=high_oscillation)
            Lambda_bar, B_bar = self._zoh(Lambda, B, dt)

        self.register_buffer('A_bar', Lambda_bar)  # (P,)
        self.register_buffer('B_bar', B_bar)  # (P, H)
        self.register_buffer('C_bar', C)  # (H, P)

        W = torch.einsum('hp,ph -> hp', self.C_bar, self.B_bar)  # (B^t .* C) (H, P)

        powers = torch.arange(kernel_size, dtype=torch.float32)  # (L,)
        V = self.A_bar.unsqueeze(-1) ** powers   # (*, L)

        kernel = torch.einsum('hp,phl->hl', W, V)  # (H, L)
        self.register_buffer('K', kernel)  # (H, L)



    @staticmethod
    def _zoh(Lambda, B, dt):
        """
        Discretize the system using the zero-order-hold transform, where A = diag(Lambda):
            A_bar = exp(A * dt)
            B_bar = (A_bar - I) * inv(A) * B
            C_bar = C
            D_bar = D
        :param Lambda: State Diagonal Matrix (Continuous System)
        :param B: Input->State Matrix (Continuous System)
        :param dt: Delta time for discretization
        :return: Lambda_bar, B_bar (Discrete System)
        """
        Lambda_bar = torch.exp(torch.einsum('ph,h->ph', Lambda, dt))
        temp = torch.einsum('ph,ph->ph', 1 / Lambda, Lambda_bar - 1)
        B_bar = torch.einsum('ph,ph->ph', temp, B)

        return Lambda_bar, B_bar

    @staticmethod
    def _normalization(Lambda, B):
        """
        Normalize the discrete SSM, where A = diag(Lambda):
            A_bar = A
            B_bar = gamma * B
            C_bar = C
            D_bar = D
        :param Lambda: State Diagonal Matrix (Discrete System)
        :param B: Input->State Matrix (Discrete System)
        :return: Lambda_bar, B_bar (Discrete System)
        """
        Zeros = torch.zeros(size=Lambda.shape, dtype=torch.float32)
        temp = torch.max(1 - torch.pow(torch.abs(Lambda), 2), Zeros)
        gamma = torch.sqrt(temp)

        B_bar = torch.einsum('ph,ph->ph', gamma, B)

        return B_bar

    def step(self, u, x=None):
        """
        Compute one time step using H independent recurrent models. Intended to be used during validation.
        Assuming that for each feature H, the diagonal discrete SSM (recurrent model) is:
            x_new = A_bar * x_old + B_bar * u_new
            y_new = C_bar * x_new, if H=1
        :param u: time step input of shape (B, H)
        :param x: time step state of shape (B, P, H)
        :return: y: time step output of shape (B, H), x: time step state of shape (B, P, H)
        """
        u = u.to(dtype=torch.complex64)  # (B, H)
        if x is None:
            x = self.x0.unsqueeze(0).expand(u.shape[0], self.B_bar.shape[0], self.B_bar.shape[1])  # (B, P, H)
        x = torch.mul(self.A_bar.unsqueeze(0), x) + torch.mul(self.B_bar.unsqueeze(0), u.unsqueeze(-2))  # (B, P, H)
        y = torch.sum(torch.mul(self.C_bar.unsqueeze(0), x.transpose(1, 2)), dim=-1)  # (B, H)

        return y, x  # (B, H), (B, P, H)

    def forward(self):
        """
        Generate the convolution kernel from the diagonal SSM parameters
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        kernel = self.K  # (H, L)

        return kernel, None
