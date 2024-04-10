import torch
from src.reservoir.state_reservoir import DiscreteStateReservoir, ContinuousStateReservoir
from src.reservoir.reservoir import Reservoir
import torch.nn as nn


class VandermondeKernel(nn.Module):
    """
    Generate convolution kernel from diagonal SSM parameters
    """
    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability, dt=None,
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
        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # Necessary condition for the Vandermonde kernel (SISO)

        input2state_reservoir = Reservoir(d_in=self.d_input, d_out=self.d_state)
        state2output_reservoir = Reservoir(d_in=self.d_state, d_out=self.d_output)

        B = input2state_reservoir.uniform_matrix(scaling=1.0, field=field)
        C = state2output_reservoir.uniform_matrix(scaling=1.0, field=field)

        if dt is None:
            state_reservoir = DiscreteStateReservoir(self.d_state)
            Lambda_bar = state_reservoir.diagonal_state_space_matrix(
                min_radius=strong_stability, max_radius=weak_stability, field=field)
            B_bar = B
        elif dt > 0:
            state_reservoir = ContinuousStateReservoir(self.d_state)
            Lambda = state_reservoir.diagonal_state_space_matrix(
                min_real_part=strong_stability, max_real_part=weak_stability, field=field)
            Lambda_bar, B_bar = self._zoh(Lambda, B, dt)
        else:
            raise ValueError("Delta time dt must be positive: set dt>0 or None for 'discrete dynamics'.")

        # Initialize parameters A, B and C
        self.A = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=True)  # (P, 2)
        self.B = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, H, 2)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (H, P, 2)

        # Register powers for Vandermonde matrix
        powers = torch.arange(kernel_size, dtype=torch.float32)
        self.register_buffer('powers', powers)  # (L)


    @staticmethod
    def _zoh(Lambda, B, dt):
        """
        Discretize the system using the zero-order-hold transform, where A = diag(Lambda):
            A_bar = exp(A * Delta)
            B_bar = (A_bar - I) * inv(A) * B
            C_bar = C
            D_bar = D
        :param Lambda: State Diagonal Matrix (Continuous System)
        :param B: Input->State Matrix (Continuous System)
        :param dt: Delta time for discretization
        :return: Lambda_bar, B_bar (Discrete System)
        """
        Ones = torch.ones(Lambda.shape[0], dtype=torch.float32)

        Lambda_bar = torch.exp(Lambda * dt)
        B_bar = torch.einsum('p,ph->ph', torch.mul(1 / Lambda, (Lambda_bar - Ones)), B)

        return Lambda_bar, B_bar

    def _construct_vandermonde(self):
        """
        Construct the Vandermonde Matrix from the diagonal state matrix A:
            vandermonde[j,l] = A[j]^l,
        where:
            j = 0, ..., d_state-1
            l = 0, ..., kernel_size-1.
        returns: vandermonde: (P,L)
        """
        A = torch.view_as_complex(self.A)
        vandermonde = A.unsqueeze(1) ** self.powers  # (P, L)
        return vandermonde

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
            x_new = A * x_old + B * u_new
            y_new = C * x_new
        :param u: time step input of shape (B, H)
        :param x: time step state of shape (B, P)
        :return: y: time step output of shape (B, H), x: time step state of shape (B, P)
        """
        u = torch.view_as_complex(u)
        A = torch.view_as_complex(self.A)  # (P)
        B = torch.view_as_complex(self.B)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)

        x = torch.einsum('p,bp->bp', A, x) + torch.einsum('ph,bh->bp', B, u)  # (B,P)
        y = torch.einsum('hp,bp->bh', C, x).real  # (B,H)

        return y, x

    def forward(self):
        """
        Generate the convolution kernel from the diagonal SSM parameters
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        vandermonde = self._construct_vandermonde()  # (P, L)
        B = torch.view_as_complex(self.B)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)

        kernel = torch.einsum('hp,pl->hl', torch.einsum('hp,ph->hp', C, B), vandermonde)  # (H, L)

        return kernel, None


class VandermondeInput2StateReservoirKernel(VandermondeKernel):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability, dt=None,
                 field='complex'):
        """
        Construct the convolution kernel with frozen A.
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
        super().__init__(d_input, d_state, kernel_size,
                         strong_stability, weak_stability, dt,
                         field)

        # Freeze B
        self.B.requires_grad_(False)


class VandermondeStateReservoirKernel(VandermondeKernel):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability, dt=None,
                 field='complex'):
        """
        Construct the convolution kernel with frozen A.
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
        super().__init__(d_input, d_state, kernel_size,
                         strong_stability, weak_stability, dt,
                         field)

        # Freeze A
        self.A.requires_grad_(False)

        # Register Vandermonde matrix for kernel computation
        vandermonde = self._construct_vandermonde()  # (P, L)
        self.register_buffer('vandermonde', vandermonde)

    def forward(self):
        """
        Generate the convolution kernel from the diagonal SSM parameters
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        B = torch.view_as_complex(self.B)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)
        kernel = torch.einsum('hp,pl->hl', torch.einsum('hp,ph->hp', C, B), self.vandermonde)  # (H, L)

        return kernel, None


class VandermondeReservoirKernel(VandermondeStateReservoirKernel):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability, dt=None,
                 field='complex'):
        """
        Construct the convolution kernel with frozen A.
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
        super().__init__(d_input, d_state, kernel_size,
                         strong_stability, weak_stability, dt,
                         field)

        # Freeze B
        self.B.requires_grad_(False)
