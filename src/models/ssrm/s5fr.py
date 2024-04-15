import torch
from src.reservoir.state import DiscreteStateReservoir, ContinuousStateReservoir
import torch.nn as nn
from src.utils.plot import plot_spectrum


"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S5FR(torch.nn.Module):
    def __init__(self, d_input, kernel_size, d_state, strong_stability, weak_stability, dt=None,
                 field='complex'):
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

        super(S5FR, self).__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input

        # Initialize parameters C and D
        C = torch.randn(self.d_output, self.d_state, dtype=torch.complex64)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (H, P, 2)
        self.D = nn.Parameter(torch.randn(self.d_output, self.d_input, dtype=torch.float32), requires_grad=True)  # (H, H)

        B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        if dt is None:
            dr = DiscreteStateReservoir(self.d_state, strong_stability, weak_stability, field)
            Lambda_bar = dr.diagonal_state_matrix()
            B_bar = B
        elif dt > 0:
            cr = ContinuousStateReservoir(self.d_state, strong_stability, weak_stability, field)
            Lambda = cr.diagonal_state_matrix()
            Lambda_bar, B_bar = self._zoh(Lambda, B, dt)  # Discretization
        else:
            raise ValueError("Delta time dt must be positive: set dt>0 otherwise 'discrete dynamics'.")

        # Initialize parameters Lambda_bar and B_bar
        self.Lambda_bar = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=False)  # Frozen Lambda_bar (P)
        self.B_bar = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, H, 2)

        # Construct the kernel
        self.kernel = nn.Parameter(self._construct_kernel(kernel_size), requires_grad=False)  # (P, L)

        # Output linear layer
        self.output_linear = nn.Sequential(nn.GELU())

    def plot_discrete_spectrum(self):
        """
        Plot the spectrum of the discrete dynamics
        :return:
        """
        plot_spectrum(self.Lambda_bar)

    @staticmethod
    def _zoh(Lambda, B, Delta):
        """
        Discretize the system using the zero-order-hold transform:
        Lambda_bar = exp(Lambda*Delta)
        B_bar = (A_bar - I) * inv(A) * B
        C = C
        D = D
        :param Lambda: State Diagonal Matrix (Continuous System)
        :param B: Input->State Matrix (Continuous System)
        :param Delta: Timestep (1 for each input dimension)
        :return: Lambda_bar, B_bar (Discrete System)
        """
        Ones = torch.ones(Lambda.shape[0], dtype=torch.float32)

        Lambda_bar = torch.exp(torch.mul(Lambda, Delta))

        B_bar = torch.einsum('p,ph->ph', torch.mul(1 / Lambda, (Lambda_bar - Ones)), B)

        return Lambda_bar, B_bar


    def _construct_kernel(self, kernel_size):
        """

        Args:
            kernel_size:

        Returns:

        """
        Lambda_bar = torch.view_as_complex(self.Lambda_bar)  # (P)
        omega_s = 2 * torch.pi * torch.fft.rfftfreq(kernel_size)  # (L//2+1)
        s = 1j * omega_s
        kernel = 1 / (s.reshape(1, -1) - Lambda_bar.unsqueeze(-1))  # (P, L)
        return kernel

    def _apply_transfer_function(self, U_s):
        """
        Apply the transfer function to the input sequence
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        # Materialize parameters
        # Lambda_bar = torch.view_as_complex(self.Lambda_bar)  # (P)
        B_bar = torch.view_as_complex(self.B_bar)  # (P,H)
        C = torch.view_as_complex(self.C)  # (H, P)

        # Construct the transfer function G(s) = C * (sI - A)^-1 * B + D = C * (Kernel) * B + D
        C_kernel = torch.einsum('hp,pl->hpl', C, self.kernel)  # (H, P, L)
        G = torch.einsum('hpl,pq->hql', C_kernel, B_bar)  # (H, H, L)

        Y_s = torch.einsum('hql,bql->bhl', G, U_s) + torch.einsum('hq,bql->bhl', self.D, U_s)  # (B, H, L)
        return Y_s

    def forward(self, u):
        """
        Forward method for the S5R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """

        U_s = torch.fft.rfftn(u)  # (B, H, L//2+1)

        # omega_s = 2 * torch.pi * torch.fft.rfftfreq(u.shape[-1])  # (L//2+1)
        # omega_s = omega_s.to(device=u.device)
        # s = 1j * omega_s

        # Compute Du part
        Y_s = self._apply_transfer_function(U_s)

        y = torch.fft.irfftn(Y_s)  # (B, H, L)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return self.output_linear(y), None
