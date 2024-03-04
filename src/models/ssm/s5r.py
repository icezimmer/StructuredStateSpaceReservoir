import torch
from src.utils.jax_compat import associative_scan
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from math import log

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""


class S5R(torch.nn.Module):
    def __init__(self, d_input, d_state, min_radius=0.9, max_radius=1, dynamics='continuous', field='complex'):
        """
        Construct an SSM model with frozen state matrix Lambda_bar:
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dynamics: 'continuous' or 'discrete'
        :param field: 'real' or 'complex'
        """
        # TODO: Delta trainable parameter not fixed to ones:
        #   Lambda_bar = Lambda_Bar(Lambda, Delta), B_bar = B(Lambda, B, Delta)

        super(S5R, self).__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input

        # Initialize parameters C and D
        self.C = nn.Parameter(torch.randn(self.d_output, self.d_state, dtype=torch.complex64))
        self.D = nn.Parameter(torch.randn(self.d_output, self.d_input, dtype=torch.float32))

        if min_radius > max_radius or min_radius < 0 or max_radius > 1:
            raise ValueError("min_radius should be less than max_radius and both should be in [0, 1].")

        # System dynamics
        if dynamics == 'continuous':
            min_real = log(min_radius)
            max_real = log(max_radius)
            Lambda = self._continuous_state_matrix(self.d_state, min_real, max_real, field)
            B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
            Delta = torch.ones(self.d_state, 1, dtype=torch.float32)  # Placeholder for future customization
            Lambda_bar, B_bar = self._zoh(Lambda, B, Delta)  # Discretization
        elif dynamics == 'discrete':
            Lambda_bar = self._discrete_state_matrix(self.d_state, min_radius, max_radius, field)
            B_bar = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        else:
            raise NotImplementedError("Dynamics must be 'continuous' or 'discrete'.")

        self.Lambda_bar = nn.Parameter(Lambda_bar, requires_grad=False)
        self.B_bar = nn.Parameter(B_bar, requires_grad=True)

        # Output linear layer
        self.output_linear = nn.Sequential(nn.GELU())

    @staticmethod
    def _discrete_state_matrix(d_state, min_radius, max_radius, field):
        """
        Create a state matrix Lambda_bar for the discrete dynamics;
        lambda = radius * (cos(theta) + i * sin(theta)):
        radius in [min_radius, max_radius),
        theta in [0, 2pi).
        :param d_state: latent state dimension
        :param field: 'complex' or 'real'
        :return: Lambda_bar
        """
        if field == 'complex':
            radius = min_radius + (max_radius - min_radius) * torch.rand(d_state, dtype=torch.float32)
            theta = 2 * torch.pi * torch.rand(d_state, dtype=torch.float32)
            alpha_tensor = radius * torch.cos(theta)
            omega_tensor = radius * torch.sin(theta)
        elif field == 'real':
            half_d_state = d_state // 2
            radius = min_radius + (max_radius - min_radius) * torch.rand(half_d_state, dtype=torch.float32)
            theta = torch.pi * torch.rand(half_d_state, dtype=torch.float32)
            alpha_tensor = torch.cat((radius * torch.cos(theta), radius * torch.cos(theta)), 0)
            omega_tensor = torch.cat((radius * torch.sin(theta), -radius * torch.sin(theta)), 0)
            if d_state % 2 == 1:
                extra_radius = min_radius + (max_radius - min_radius) * torch.rand(1, dtype=torch.float32)
                # Choose 0 or pi randomly for extra_theta
                extra_theta = torch.randint(0, 2, (1,)) * torch.pi
                alpha_tensor = torch.cat((alpha_tensor, extra_radius * torch.cos(extra_theta)), 0)
                omega_tensor = torch.cat((omega_tensor, extra_radius * torch.sin(extra_theta)), 0)
        else:
            raise NotImplementedError("The field must be 'complex' or 'real'.")

        Lambda_bar = torch.complex(alpha_tensor, omega_tensor)
        return Lambda_bar.view(-1, 1)

    @staticmethod
    def _continuous_state_matrix(d_state, min_real, max_real, field):
        """
        Create a state matrix Lambda for the continuous dynamics;
        lambda = log(radius) + i * theta:
        Re(lambda) in [min_real, max_real) = [log(min_radius), log(max_radius)),
        Im(lambda) in [0, 2pi).
        :param d_state: latent state dimension
        :param field: 'complex' or 'real'
        :return: Lambda
        """
        if field == 'complex':
            real_tensor = min_real + (max_real - min_real) * torch.rand(d_state, dtype=torch.float32)
            imag_tensor = 2 * torch.pi * torch.rand(d_state, dtype=torch.float32)
        elif field == 'real':
            half_d_state = d_state // 2
            real_tensor = min_real + (max_real - min_real) * torch.rand(half_d_state, dtype=torch.float32)
            imag_tensor = torch.pi * torch.rand(half_d_state, dtype=torch.float32)
            real_tensor = torch.cat((real_tensor, real_tensor), 0)
            imag_tensor = torch.cat((imag_tensor, -imag_tensor), 0)
            if d_state % 2 == 1:
                extra_real = min_real + (max_real - min_real) * torch.rand(1, dtype=torch.float32)
                # Choose 0 or pi randomly for extra_imag (extra_theta)
                extra_imag = torch.randint(0, 2, (1,)) * torch.pi
                real_tensor = torch.cat((real_tensor, extra_real), 0)
                imag_tensor = torch.cat((imag_tensor, extra_imag), 0)
        else:
            raise NotImplementedError("The field must be 'complex' or 'real'.")

        Lambda = torch.complex(real_tensor, imag_tensor)
        return Lambda.view(-1, 1)

    def plot_discrete_spectrum(self):
        """
        Plot the spectrum of the discrete dynamics
        :return:
        """
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

    def _apply_scan(self, input_sequence):
        """
        Apply the SSM to the single input sequence
        :param input_sequence: tensor of shape (H,L) = (d_input, input_length)
        :return: convolution: tensor of shape (P,L) = (d_state, input_length)
        """
        complex_input_sequence = input_sequence.to(self.Lambda_bar.dtype)  # Cast to correct complex type

        # Time Invariant B(t) = B of shape (P,H)
        Bu_elements = torch.mm(self.B_bar, complex_input_sequence)  # Tensor of shape (P,L)

        Lambda_elements = self.Lambda_bar.tile(1, input_sequence.shape[1])  # Tensor of shape (P,L)

        # convolution, resulting tensor of shape (P,L)
        _, convolution = associative_scan(S5R.binary_operator, (Lambda_elements, Bu_elements), axis=1)

        # Result of shape (P,L)
        return convolution

    def forward(self, u):
        """
        Forward method for the S5R model
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        A_Bu = torch.vmap(self._apply_scan)(u)
        C_A_Bu = torch.einsum('hp,bpl->bhl', self.C, A_Bu).real

        # Compute Du part
        Du = torch.einsum('hh,bhl->bhl', self.D, u)

        # Return a dummy state to satisfy this repo's interface, but this can be modified
        return self.output_linear(C_A_Bu + Du), None
