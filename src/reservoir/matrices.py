import torch


class ReservoirMatrix:
    def __init__(self, d_in, d_out):
        """
        Generate random matrices.
        """
        self.d_in = d_in
        self.d_out = d_out

    def uniform_ring(self, min_radius, max_radius, field):
        """
        Create the random uniform matrix such that each matrix values: -radius = |W_ij| = radius.
        :param min_radius: float, the minimum radius of the matrix values
        :param max_radius: float, the maximum radius of the matrix values
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        if min_radius < 0:
            raise ValueError('The minimum radius must be non-negative')
        if max_radius < min_radius:
            raise ValueError('The maximum radius must be greater or equal to the minimum radius')

        radius = (min_radius + (max_radius - min_radius) *
                  torch.sqrt(torch.rand(self.d_out, self.d_in, dtype=torch.float32)))

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
