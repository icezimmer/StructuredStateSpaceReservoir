import torch
from src.utils.jax_compat import associative_scan
import torch.nn as nn

"""
Define a model subclass of torch.nn.Module that implements a linear time-invariant system.
"""


class SSM(torch.nn.Module):
    def __init__(self, d_input, eigs, delta_min, delta_max):
        # x_new = Lambda_bar * x_old + B_bar * u_new
        # y_new = C * x_new + D * u_new

        super(SSM, self).__init__()

        #self.Lambda = Lambda
        self.Lambda = SSM.__state_matrix(eigs)

        self.d_input = d_input
        self.d_state = self.Lambda.shape[0]
        self.d_output = self.d_input

        self.B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        self.C = torch.randn(self.d_output, self.d_state, dtype=torch.complex64)
        self.D = torch.randn(self.d_output, self.d_input, dtype=torch.float32)

        self.Delta = delta_min + (delta_max - delta_min) * torch.rand(self.d_state, 1, dtype=torch.float32)
        self.Lambda_bar = self.__state_matrix(eigs)
        self.Lambda_bar, self.B_bar = self.__zoh()  # Discretize once during initialization

        self.x = torch.zeros(self.d_state, 1, dtype=torch.float32)

        self.output_linear = nn.Sequential(
            nn.GELU(),
        )


    @staticmethod
    def __state_matrix(eigs):
        """
        Compute the state matrix of the system.
        """
        d_state = sum(mul * 2 for mul in eigs.values())
        Lambda = torch.empty(d_state, dtype=torch.complex64)

        index = 0
        for (alpha, omega), mul in eigs.items():
            # Create tensors for the real and imaginary parts
            alpha_tensor = torch.full((mul,), alpha, dtype=torch.float32)
            omega_tensor = torch.full((mul,), omega, dtype=torch.float32)

            # Construct the complex numbers and their conjugates
            Lambda[index:index + mul] = torch.complex(alpha_tensor, omega_tensor)
            Lambda[index + mul:index + 2 * mul] = torch.complex(alpha_tensor, -omega_tensor)

            index += 2 * mul

        # Reshape Lambda to match the expected output shape
        Lambda = Lambda.view(-1, 1)
        return Lambda

    def __zoh(self):
        """
        Discretize the system using the zero-order-hold transform.
        """
        # A_bar = exp(A*Delta)
        # B = (A_bar - I) * inv(A) * B
        # C = C
        # D = D
        Ones = torch.ones(self.d_state, 1, dtype=torch.float32)
        Lambda_bar = torch.exp(torch.mul(self.Lambda, self.Delta))
        B_bar = torch.mul(torch.mul(1 / self.Lambda, (Lambda_bar - Ones)), self.B)

        return Lambda_bar, B_bar

    @staticmethod
    def binary_operator(element_i, element_j):
        """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
        element_i: tuple containing A_i and Bu_i at position i
        (P,), (P,)
        element_j: tuple containing A_j and Bu_j at position j
        (P,), (P,)
        Returns:
        new element ( A_out, Bu_out )
        """
        A_i, Bu_i = element_i
        A_j, Bu_j = element_j

        # Check for empty tensors and handle accordingly
        if A_i.nelement() == 0 or A_j.nelement() == 0:
            # Assuming returning zeros
            shape = (max(A_i.size(0), A_j.size(0)), max(A_i.size(1), A_j.size(1)))
            return torch.zeros(shape), torch.zeros(shape)

        # Proceed with original operation if tensors are not empty
        A_out = torch.mul(A_j, A_i)
        Bu_out = torch.mul(A_j, Bu_i + Bu_j)

        return A_out, Bu_out

    def __apply_ssm(self, input_sequence):
        """
        Apply the SSM to the single input sequence
        Args:
         input_sequence: tensor of shape (H,L) = (d_input, length)
        Returns:
        """
        complex_input_sequence = input_sequence.to(self.Lambda_bar.dtype)  # Cast to correct complex type

        # Time Invariant B(t) = B of shape (P,H)
        Bu_elements = torch.mm(self.B_bar, complex_input_sequence)  # Tensor of shape (P,L)
        print("Bu_elements", Bu_elements.shape)

        Lambda_elements = self.Lambda_bar.tile(1, input_sequence.shape[1])  # Tensor of shape (P,L)
        print("Lambda_elements", Lambda_elements.shape)

        # xs of shape (P,L)
        _, xs = associative_scan(SSM.binary_operator, (Lambda_elements.transpose(0, 1), Bu_elements.transpose(0, 1)))
        xs = xs.transpose(0, 1)
        print("xs", xs.shape)

        # Du of shape (H,L)
        Du = torch.mm(self.D, input_sequence)
        print("Du", Du.shape)

        # TODO: the last element of xs (non-bidir) is the hidden state, allow returning it
        #return torch.vmap(lambda x: (torch.mm(self.C, x).real)(xs) + Du

        return self.output_linear(torch.mm(self.C, xs) + Du)

    def forward(self, u):
        return torch.vmap(self.__apply_ssm)(u)
