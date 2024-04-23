import warnings
import torch


class Reservoir:
    def __init__(self, d_in, d_out):
        """
        Generate the discrete state matrix.
        """
        self.d_in = d_in
        self.d_out = d_out

    def uniform_disk_matrix(self, radius, field):
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

    def uniform_ring_matrix(self, radius, field):
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

    def single_value_matrix(self, value):
        if isinstance(value, complex):
            value = torch.tensor(value, dtype=torch.complex64)
        else:
            value = torch.tensor(value, dtype=torch.float32)

        W = value * torch.ones(self.d_out, self.d_in, dtype=value.dtype)

        return W


class StructuredReservoir(Reservoir):
    def __init__(self, d_in, d_out):
        """
        Generate the discrete state matrix.
        """

        if d_in < 1:
            raise ValueError("Input dimension must be an integer > 0.")

        if d_out < d_in:
            raise ValueError("Output dimension must be an integer >= input dimension.")

        if d_in == 1:
            warnings.warn("Input dimension is equal to 1: the structured reservoir is not effective.")
        if d_out == d_in:
            warnings.warn("Input dimension is equal to output dimension: the structured reservoir is not effective.")

        super().__init__(d_in, d_out)

        self.M = self._mask()

    # TODO: repair mask dimension
    def _mask(self):
        if self.d_in == 1:
            return torch.ones(self.d_out, self.d_in)

        if self.d_out == self.d_in:
            return torch.eye(self.d_in)

        bits = [torch.tensor([0, 1]) for _ in range(self.d_in)]
        all_combinations = torch.cartesian_prod(*bits)
        non_zero_filter = all_combinations.sum(dim=1) != 0
        mask = all_combinations[non_zero_filter]
        num_repeats = (self.d_out + self.d_in - 1) // self.d_in

        return mask.repeat(num_repeats, 1)[:self.d_out, :]

    def uniform_disk_matrix(self, radius, field):
        """
        Create the random uniform matrix such that each matrix values: -radius <= |W_ij| < radius.
        :param radius: float, the radius of the matrix
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        W = super().uniform_disk_matrix(radius, field)
        W = torch.einsum('ph,ph -> ph', self.M, W)
        return W

    def uniform_ring_matrix(self, radius, field):
        W = super().uniform_ring_matrix(radius, field)
        W = torch.einsum('ph,ph -> ph', self.M, W)
        return W

    def single_value_matrix(self, value):
        W = super().single_value_matrix(value)
        W = torch.einsum('ph,ph -> ph', self.M, W)
        return W
