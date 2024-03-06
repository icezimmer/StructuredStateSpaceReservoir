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


class S5R(torch.nn.Module):
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

        super(S5R, self).__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input

        # Initialize parameters C and D
        C = torch.randn(self.d_output, self.d_state, dtype=torch.complex64)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (H, P, 2)
        D = torch.randn(self.d_output, self.d_input, dtype=torch.float32)
        self.D = nn.Parameter(D, requires_grad=True)  # (H, H)

        B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        if dynamics == 'discrete':
            if high_stability > low_stability or high_stability < 0 or low_stability > 1:
                raise ValueError("For the discrete dynamics stability we must have: "
                                 "0 <= 'high_stability' < |lambda| <= 'low_stability' <= 1.")
            else:
                self.high_stability = high_stability
                self.low_stability = low_stability
                cr = ContinuousStateReservoir(self.d_state, self.high_stability, self.low_stability, field)
                Lambda = cr.diagonal_state_matrix()
                Delta = torch.ones(self.d_state, dtype=torch.float32)  # Placeholder for future customization
                Lambda_bar, B_bar = self._zoh(Lambda, B, Delta)  # Discretization
        elif dynamics == 'continuous':
            if high_stability > low_stability or low_stability > 0:
                raise ValueError("For the continuous dynamics stability we must have: "
                                 "'high_stability' < Re(lambda) <= 'low_stability' <= 0.")
            else:
                self.high_stability = high_stability
                self.low_stability = low_stability
                dr = DiscreteStateReservoir(self.d_state, self.high_stability, self.low_stability, field)
                Lambda_bar = dr.diagonal_state_matrix()
                B_bar = B
        else:
            raise ValueError("Dynamics must be 'continuous' or 'discrete'.")
        self.dynamics = dynamics

        # Initialize parameters Lambda_bar and B_bar
        self.Lambda_bar = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=False)  # Frozen Lambda_bar (P)
        self.B_bar = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, H, 2)

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

    @staticmethod
    def binary_operator(element_i, element_j):
        """
        Binary associative operator for the associative scan
        :param element_i: tuple A_i, Bu_i
        :param element_j: tuple A_j, Bu_j
        :return: A_j * A_i, A_j * Bu_i + Bu_j
        """
        A_i, Bu_i = element_i
        A_j, Bu_j = element_j

        # Proceed with original operation if tensors are not empty
        A_power = torch.mul(A_j, A_i)
        convolution = torch.mul(A_j, Bu_i) + Bu_j

        return A_power, convolution

    def _apply_scan(self, input_sequence):
        """
        Apply the SSM to the single input sequence
        :param input_sequence: tensor of shape (H,L) = (d_input, input_length)
        :return: convolution: tensor of shape (P,L) = (d_state, input_length)
        """
        # Materialize parameters
        Lambda_bar = torch.view_as_complex(self.Lambda_bar)  # (P)
        B_bar = torch.view_as_complex(self.B_bar)  # (P,H)
        complex_input_sequence = input_sequence.to(Lambda_bar.dtype)  # Cast to correct complex type

        # Time Invariant B(t) = B
        Bu_elements = torch.mm(B_bar, complex_input_sequence)  # (P, L)

        Lambda_elements = repeat(Lambda_bar, 'p -> p l', l=input_sequence.shape[1])

        # Compute convolution
        _, convolution = associative_scan(S5R.binary_operator, (Lambda_elements, Bu_elements), axis=1)  # (P, L)

        return convolution

    def forward(self, u):
        """
        Forward method for the S5R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        # Materialize parameters
        C = torch.view_as_complex(self.C)

        A_Bu = torch.vmap(self._apply_scan)(u)
        C_A_Bu = torch.einsum('hp,bpl->bhl', C, A_Bu).real

        # Compute Du part
        Du = torch.einsum('hh,bhl->bhl', self.D, u)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return self.output_linear(C_A_Bu + Du), None
