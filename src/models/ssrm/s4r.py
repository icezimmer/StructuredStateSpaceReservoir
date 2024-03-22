import torch
from src.reservoir.state_reservoir import DiscreteStateReservoir, ContinuousStateReservoir
import torch.nn as nn
from src.utils.plot import plot_spectrum


"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S4R(torch.nn.Module):
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

        super(S4R, self).__init__()

        self.d_input = d_input
        self.kernel_size = kernel_size
        self.d_state = d_state
        self.d_output = self.d_input

        # Initialize parameters C and D
        C = torch.randn(self.d_state, dtype=torch.complex64)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (P, 2)
        self.D = nn.Parameter(torch.randn(self.d_input, dtype=torch.float32), requires_grad=True)  # (H)

        B = torch.randn(self.d_state, dtype=torch.complex64)
        if dt is None:
            dr = DiscreteStateReservoir(self.d_state, strong_stability, weak_stability, field)
            Lambda_bar = dr.diagonal_state_matrix()
            B_bar = B
        elif dt > 0:
            cr = ContinuousStateReservoir(self.d_state, strong_stability, weak_stability, field)
            Lambda = cr.diagonal_state_matrix()
            # Delta = torch.ones(self.d_state, dtype=torch.float32)
            Lambda_bar, B_bar = self._zoh(Lambda, B, dt)  # Discretization
        else:
            raise ValueError("Delta time dt must be positive: set dt>0 otherwise 'discrete dynamics'.")

        # Initialize parameters Lambda_bar and B_bar
        self.Lambda_bar = nn.Parameter(Lambda_bar, requires_grad=False)  # Frozen Lambda_bar (P)
        self.B_bar = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, 2)

        # Construct the kernel
        self.vandermonde = nn.Parameter(self._construct_vandermonde(), requires_grad=False)  # (P, L)

        # Output linear layer
        self.non_linearity = nn.Sequential(nn.Tanh())

    def plot_discrete_spectrum(self):
        """
        Plot the spectrum of the discrete dynamics
        :return:
        """
        plot_spectrum(self.Lambda_bar)

    @staticmethod
    def _zoh(Lambda, B, dt):
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

        # Lambda_bar = torch.exp(torch.mul(Lambda, Delta))
        Lambda_bar = torch.exp(Lambda * dt)

        B_bar = torch.einsum('p,p->p', torch.mul(1 / Lambda, (Lambda_bar - Ones)), B)

        return Lambda_bar, B_bar

    def _construct_vandermonde(self):
        """
        Args:
            kernel_size:

        Returns:

        """
        vandermonde = self.Lambda_bar.unsqueeze(1) ** torch.arange(self.kernel_size, dtype=torch.float32)  # (P, L)
        return vandermonde

    def _apply_kernel(self, u):
        """
        Apply the transfer function to the input sequence
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        B_bar = torch.view_as_complex(self.B_bar)  # (P)
        C = torch.view_as_complex(self.C)  # (P)

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)

        kernel = torch.einsum('p,pl->l', torch.einsum('p,p->p', C, B_bar), self.vandermonde)  # (L)
        kernel_s = torch.fft.fft(kernel, dim=-1)

        y = torch.fft.ifft(torch.einsum('bhl,l->bhl', u_s, kernel_s), dim=-1)  # (B, H, L)

        return y

    def forward(self, u):
        """
        Forward method for the S5R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """

        # Compute Du part
        y = self._apply_kernel(u)
        y = y.real + torch.einsum('h,bhl->bhl', self.D, u)  # (B, H, L), self.D.unsqueeze(-1) is (H, 1)

        y = self.non_linearity(y)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return y, None
