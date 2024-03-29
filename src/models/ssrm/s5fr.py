import torch
from src.reservoir.state_reservoir import DiscreteStateReservoir, ContinuousStateReservoir
from src.utils.jax_compat import associative_scan
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from einops import repeat

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


class S5FR(torch.nn.Module):
    def __init__(self, d_input, d_state, high_stability, low_stability, dynamics, field='complex'):
        """
        Construct an SSM model with frozen state matrix Lambda_bar:
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dynamics: 'continuous' or 'discrete'
        :param field: 'real' or 'complex'
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
        D = torch.randn(self.d_output, self.d_input, dtype=torch.complex64)
        self.D = nn.Parameter(torch.view_as_real(D), requires_grad=True)  # (H, H)

        B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        if dynamics == 'discrete':
            if high_stability > low_stability or high_stability < 0 or low_stability > 1:
                raise ValueError("For the discrete dynamics stability we must have: "
                                 "0 <= 'high_stability' < |lambda| <= 'low_stability' <= 1.")
            else:
                self.high_stability = high_stability
                self.low_stability = low_stability
                dr = DiscreteStateReservoir(self.d_state, self.high_stability, self.low_stability, field)
                Lambda_bar = dr.diagonal_state_matrix()
                B_bar = B
        elif dynamics == 'continuous':
            if high_stability > low_stability or low_stability > 0:
                raise ValueError("For the continuous dynamics stability we must have: "
                                 "'high_stability' < Re(lambda) <= 'low_stability' <= 0.")
            else:
                self.high_stability = high_stability
                self.low_stability = low_stability
                cr = ContinuousStateReservoir(self.d_state, self.high_stability, self.low_stability, field)
                Lambda = cr.diagonal_state_matrix()
                Delta = torch.ones(self.d_state, dtype=torch.float32)  # Placeholder for future customization
                Lambda_bar, B_bar = self._zoh(Lambda, B, Delta)  # Discretization
        else:
            raise ValueError("Dynamics must be 'continuous' or 'discrete'.")
        self.dynamics = dynamics

        # Initialize parameters Lambda_bar and B_bar
        self.Lambda_bar = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=False)  # Frozen Lambda_bar (P)
        self.B_bar = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, H, 2)

        # tensor with values from e^(-j*2*pi) to e^(-j*2*pi/L//2+1)
        self.L = torch.arange(0, d_input // 2 + 1, dtype=torch.float32)


        # Output linear layer
        self.output_linear = nn.Sequential(nn.GELU())

    def plot_discrete_spectrum(self):
        """
        Plot the spectrum of the discrete dynamics
        :return:
        """
        # Extracting real and imaginary parts
        Lambda_bar = self.Lambda_bar.clone().detach()
        real_parts = Lambda_bar[:, 0]
        imaginary_parts = Lambda_bar[:, 1]

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.scatter(real_parts, imaginary_parts, color='red', marker='o')
        plt.title('Complex Eigs (Discrete Dynamics)')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        # Adding a unit circle
        circle = Circle((0, 0), 1, fill=False, color='blue', linestyle='--')
        plt.gca().add_patch(circle)

        plt.grid(True)
        plt.axhline(y=0, color='k')  # Adds x-axis
        plt.axvline(x=0, color='k')  # Adds y-axis
        plt.show()

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

    def _apply_transfer_function(self, U_s, s):
        """
        Apply the transfer function to the input sequence
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        # Materialize parameters
        Lambda_bar = torch.view_as_complex(self.Lambda_bar)  # (P)
        B_bar = torch.view_as_complex(self.B_bar)  # (P,H)
        C = torch.view_as_complex(self.C)  # (H, P)
        D = torch.view_as_complex(self.D)  # (H, H)

        # Construct the transfer function G(s) = C * (sI - A)^-1 * B + D
        mid = 1 / (s.reshape(1, -1) - Lambda_bar.unsqueeze(-1))  # (P, L)
        C_mid = torch.einsum('hp,pl->hpl', C, mid)  # (H, P, L)
        G = torch.einsum('hpl,pq->hql', C_mid, B_bar)  # (H, H, L)

        Y_s = torch.einsum('hql,bql->bhl', G, U_s) + torch.einsum('hq,bql->bhl', D, U_s)  # (B, H, L)
        return Y_s

    def forward(self, u):
        """
        Forward method for the S5R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """

        U_s = torch.fft.rfftn(u)  # (B, H, L//2+1)

        omega_s = 2 * torch.pi * torch.fft.rfftfreq(u.shape[-1])  # (L//2+1)
        s = 1j * omega_s

        # Compute Du part
        Y_s = self._apply_transfer_function(U_s, s)

        y = torch.fft.irfftn(Y_s)  # (B, H, L)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return self.output_linear(y), None
