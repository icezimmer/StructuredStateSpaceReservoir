import torch
from src.utils.jax_compat import associative_scan
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""


class SSM(torch.nn.Module):
    def __init__(self, d_input, d_state, dynamics='continuous', field='complex'):
        """
        SSM model
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dynamics: 'continuous' or 'discrete'
        :param field: 'real' or 'complex'
        """
        # TODO: Delta trainable parameter not fixed to ones:
        #   Lambda_bar = Lambda_Bar(Lambda, Delta), B_bar = B(Lambda, B, Delta)

        super(SSM, self).__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input

        # Initialize parameters C and D
        self.C = nn.Parameter(torch.randn(self.d_output, self.d_state, dtype=torch.complex64))
        self.D = nn.Parameter(torch.randn(self.d_output, self.d_input, dtype=torch.float32))

        # System dynamics
        if dynamics == 'continuous':
            Lambda = self._continuous_state_matrix(self.d_state, field)
            B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
            Delta = torch.ones(self.d_state, 1, dtype=torch.float32)  # Placeholder for future customization
            Lambda_bar, B_bar = self._zoh(Lambda, B, Delta)  # Discretization
        elif dynamics == 'discrete':
            Lambda_bar = self._discrete_state_matrix(self.d_state, field)
            B_bar = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        else:
            raise NotImplementedError("Dynamics must be 'continuous' or 'discrete'.")

        self.Lambda_bar = nn.Parameter(Lambda_bar, requires_grad=False)
        self.B_bar = nn.Parameter(B_bar, requires_grad=True)

        # Output linear layer
        self.output_linear = nn.Sequential(nn.GELU())

    @staticmethod
    def _discrete_state_matrix(d_state, field):
        if field == 'complex':
            radius = torch.rand(d_state, dtype=torch.float32)
            theta = 2 * torch.pi * torch.rand(d_state, dtype=torch.float32)
            alpha_tensor = radius * torch.cos(theta)
            omega_tensor = radius * torch.sin(theta)
        elif field == 'real':
            half_d_state = d_state // 2
            radius = torch.rand(half_d_state)
            theta = torch.pi * torch.rand(half_d_state)
            alpha_tensor = torch.cat((radius * torch.cos(theta), radius * torch.cos(theta)), 0)
            omega_tensor = torch.cat((radius * torch.sin(theta), -radius * torch.sin(theta)), 0)
            if d_state % 2 == 1:
                extra_radius = torch.rand(1, dtype=torch.float32)
                # Choose 0 or pi randomly for extra_theta
                extra_theta = torch.randint(0, 2, (1,)) * torch.pi
                alpha_tensor = torch.cat((alpha_tensor, extra_radius * torch.cos(extra_theta)), 0)
                omega_tensor = torch.cat((omega_tensor, extra_radius * torch.sin(extra_theta)), 0)
        else:
            raise NotImplementedError("The field must be 'complex' or 'real'.")

        Lambda = torch.complex(alpha_tensor, omega_tensor)
        return Lambda.view(-1, 1)

    @staticmethod
    def _continuous_state_matrix(d_state, field):
        if field == 'complex':
            real_tensor = -torch.rand(d_state, dtype=torch.float32)  # Re(lambda) in (-1, 0]
            imag_tensor = 2 * torch.pi * torch.rand(d_state, dtype=torch.float32)  # Im(lambda) in [0, 2pi)
        elif field == 'real':
            half_d_state = d_state // 2
            real_tensor = -torch.rand(half_d_state, dtype=torch.float32)  # Re(lambda) in (-1, 0]
            imag_tensor = torch.pi * torch.rand(half_d_state, dtype=torch.float32)  # Im(lambda) in [0, pi)
            real_tensor = torch.cat((real_tensor, real_tensor), 0)
            imag_tensor = torch.cat((imag_tensor, -imag_tensor), 0)
            if d_state % 2 == 1:
                extra_real = -torch.rand(1, dtype=torch.float32)
                extra_imag = torch.zeros(1, dtype=torch.float32)
                real_tensor = torch.cat((real_tensor, extra_real), 0)
                imag_tensor = torch.cat((imag_tensor, extra_imag), 0)
        else:
            raise NotImplementedError("The field must be 'complex' or 'real'.")

        Lambda = torch.complex(real_tensor, imag_tensor)
        return Lambda.view(-1, 1)

    def plot_discrete_spectrum(self):
        # Extracting real and imaginary parts
        Lambda_bar = self.Lambda_bar.clone().detach()
        real_parts = Lambda_bar.real
        imaginary_parts = Lambda_bar.imag

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
        Ones = torch.ones(Lambda.shape[0], 1, dtype=torch.float32)

        Lambda_bar = torch.exp(torch.mul(Lambda, Delta))
        B_bar = torch.mul(torch.mul(1 / Lambda, (Lambda_bar - Ones)), B)

        return Lambda_bar, B_bar

    @staticmethod
    def binary_operator(element_i, element_j):
        """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
        element_i: tuple containing A^i and Bu_i
        (P,), (P,)
        element_j: tuple containing A^j and Bu_j
        (P,), (P,)
        Returns:
        new element ( A^(i+j), convolution )
        """
        A_i, Bu_i = element_i
        A_j, Bu_j = element_j

        # Proceed with original operation if tensors are not empty
        A_power = torch.mul(A_j, A_i)
        convolution = torch.mul(A_j, Bu_i) + Bu_j

        return A_power, convolution

    # def _apply_ssm(self, input_sequence):
    #     """
    #     Apply the SSM to the single input sequence
    #     Args:
    #      input_sequence: tensor of shape (H,L) = (d_input, length)
    #     Returns:
    #     """
    #     complex_input_sequence = input_sequence.to(self.Lambda_bar.dtype)  # Cast to correct complex type
    #
    #     # Time Invariant B(t) = B of shape (P,H)
    #     Bu_elements = torch.mm(self.B_bar, complex_input_sequence)  # Tensor of shape (P,L)
    #
    #     Lambda_elements = self.Lambda_bar.tile(1, input_sequence.shape[1])  # Tensor of shape (P,L)
    #
    #     # xs of shape (P,L)
    #     _, xs = associative_scan(SSM.binary_operator,
    #                              (Lambda_elements.transpose(0, 1), Bu_elements.transpose(0, 1)))
    #     xs = xs.transpose(0, 1)
    #
    #     # Du of shape (H,L)
    #     Du = torch.mm(self.D, input_sequence)
    #
    #     # TODO: the last element of xs (non-bidir) is the hidden state, allow returning it
    #     #return torch.vmap(lambda x: (torch.mm(self.C, x).real)(xs) + Du
    #     #print("torch.vmap(lambda x: torch.mm(self.C, x).real)(xs)", torch.vmap(lambda x: torch.mm(self.C, x).real)(xs).shape)
    #
    #     # result of shape (H,L)
    #     return self.output_linear(torch.mm(self.C, xs).real + Du)

    def _apply_ssm(self, input_sequence):
        """
        Apply the SSM to the single input sequence
        Args:
         input_sequence: tensor of shape (H,L) = (d_input, length)
        Returns:
        """
        complex_input_sequence = input_sequence.to(self.Lambda_bar.dtype)  # Cast to correct complex type

        # Time Invariant B(t) = B of shape (P,H)
        Bu_elements = torch.mm(self.B_bar, complex_input_sequence)  # Tensor of shape (P,L)

        Lambda_elements = self.Lambda_bar.tile(1, input_sequence.shape[1])  # Tensor of shape (P,L)

        # xs of shape (P,L)
        _, xs = associative_scan(SSM.binary_operator,
                                 (Lambda_elements.transpose(0, 1), Bu_elements.transpose(0, 1)))
        xs = xs.transpose(0, 1)

        # result of shape (H,L)
        return xs

    def forward(self, u):
        A_Bu = torch.vmap(self._apply_ssm)(u)
        C_A_Bu = torch.einsum('hp,bpl->bhl', self.C, A_Bu).real

        # Compute Du part
        Du = torch.einsum('hh,bhl->bhl', self.D, u)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return self.output_linear(C_A_Bu + Du), None
