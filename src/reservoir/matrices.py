import warnings
import torch


class ReservoirMatrix:
    def __init__(self, d_in, d_out):
        """
        Generate random matrices.
        """
        self.d_in = d_in
        self.d_out = d_out

    def uniform_disk(self, radius, field):
        """
        Create the random uniform matrix such that each matrix values: -radius <= |W_ij| < radius.
        :param radius: float, the radius of the matrix
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        if radius < 0:
            raise ValueError('The radius must be non-negative')

        if field == 'real':
            W = - radius + 2 * radius * torch.rand(self.d_out, self.d_in, dtype=torch.float32)
        elif field == 'complex':
            theta = 2 * torch.pi * torch.rand(self.d_out, self.d_in, dtype=torch.float32)
            radius = radius * torch.sqrt(torch.rand(self.d_out, self.d_in, dtype=torch.float32))

            real_part = radius * torch.cos(theta)
            imag_part = radius * torch.sin(theta)

            W = torch.complex(real_part, imag_part)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        return W

    def uniform_ring(self, radius, field):
        """
        Create the random uniform matrix such that each matrix values: -radius = |W_ij| = radius.
        :param radius: float, the radius of the matrix
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        if radius < 0:
            raise ValueError('The radius must be non-negative')

        if field == 'real':
            random_signs = torch.sign(torch.randn(self.d_out, self.d_in))  # -1 or 1
            W = radius * random_signs
        elif field == 'complex':
            theta = 2 * torch.pi * torch.rand(self.d_out, self.d_in, dtype=torch.float32)
            real_part = radius * torch.cos(theta)
            imag_part = radius * torch.sin(theta)

            W = torch.complex(real_part, imag_part)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        return W

    def single_value(self, value):
        if isinstance(value, complex):
            value = torch.tensor(value, dtype=torch.complex64)
        else:
            value = torch.tensor(value, dtype=torch.float32)

        W = value * torch.ones(self.d_out, self.d_in, dtype=value.dtype)

        return W
