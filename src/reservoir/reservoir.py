import torch


class Reservoir:
    def __init__(self, d_in, d_out):
        """
        Generate the discrete state matrix.
        """
        self.d_in = d_in
        self.d_out = d_out

    def uniform_matrix(self, scaling, field):
        """
        Create the random uniform matrix such that each matrix value |W_ij| < scaling.
        param:
            scaling: float, the scaling of the matrix
            field: str, the field of the matrix, 'real' or 'complex'
        """
        if scaling < 0:
            raise ValueError('The input scaling must be non-negative')

        if field == 'real':
            W = - scaling + 2 * scaling * torch.rand(self.d_out, self.d_in, dtype=torch.float32)
        elif field == 'complex':
            theta = 2 * torch.pi * torch.rand(self.d_out, self.d_in, dtype=torch.float32)
            radius = scaling * torch.sqrt(torch.rand(self.d_out, self.d_in, dtype=torch.float32))

            real_part = radius * torch.cos(theta)
            imag_part = radius * torch.sin(theta)

            W = torch.complex(real_part, imag_part)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        return W
