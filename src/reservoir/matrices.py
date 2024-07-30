import warnings
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
        Create the random uniform matrix such that each matrix values:
        min_radius <= |W_ij| <= max_radius.
        :param min_radius: float, the minimum radius of the matrix values
        :param max_radius: float, the maximum radius of the matrix values
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        if min_radius < 0.0:
            raise ValueError('The minimum radius must be non-negative')
        if max_radius < min_radius:
            raise ValueError('The maximum radius must be greater or equal to the minimum radius')

        if field == 'real':
            radius = (min_radius + (max_radius - min_radius) *
                      torch.rand(self.d_out, self.d_in, dtype=torch.float32))
        elif field == 'complex':
            radius = (min_radius + (max_radius - min_radius) *
                      torch.sqrt(torch.rand(self.d_out, self.d_in, dtype=torch.float32)))
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

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

    def uniform_interval(self, min_value, max_value):
        """
        Create the random real uniform matrix such that each matrix values:
        min_value <= W_ij <= max_value.
        :param min_radius: float, the minimum radius of the matrix values
        :param max_radius: float, the maximum radius of the matrix values
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        if max_value < min_value:
            raise ValueError('The maximum value must be greater or equal to the minimum value')

        W = min_value + (max_value - min_value) * torch.rand(self.d_out, self.d_in, dtype=torch.float32)

        return W

    def uniform_slice(self, min_radius, max_radius, min_theta, max_theta):
        """
        Create the random complex uniform matrix such that each matrix values:
        min_radius <= abs(W_ij) <= max_radius;
        min_theta <= angle(W_ij) <= max_theta.
        :param min_radius: float, the minimum radius of the matrix values
        :param max_radius: float, the maximum radius of the matrix values
        :param field: str, the field of the matrix, 'real' or 'complex'
        """
        if min_radius < 0.0:
            raise ValueError('The minimum radius must be non-negative')
        if max_radius < min_radius:
            raise ValueError('The maximum radius must be greater or equal to the minimum radius')
        if min_radius < 0.0:
            raise ValueError('The minimum radius must be non-negative')
        if max_radius < min_radius:
            raise ValueError('The maximum radius must be greater or equal to the minimum radius')

        if min_theta > max_theta or min_theta < 0.0:
            raise ValueError("For the discrete dynamics we must have:"
                             "0 <= min_theta <= theta < max_theta.")
        if max_theta > 2 * torch.pi:
            warnings.warn("Being max_theta > 2pi, max_theta is clipped to 2pi")
            max_theta = 2 * torch.pi

        radius = (min_radius + (max_radius - min_radius) *
                  torch.sqrt(torch.rand(self.d_out, self.d_in, dtype=torch.float32)))

        theta = min_theta + (max_theta - min_theta) * torch.rand(self.d_out, self.d_in, dtype=torch.float32)
        real_part = radius * torch.cos(theta)
        imag_part = radius * torch.sin(theta)

        W = torch.complex(real_part, imag_part)

        return W

    def convex_combination(self):
        W = torch.rand(self.d_out, self.d_in, dtype=torch.float32)

        row_sums = W.sum(dim=1, keepdim=True)
        W = W / row_sums

        return W

    def single_value(self, value):
        if isinstance(value, complex):
            value = torch.tensor(value, dtype=torch.complex64)
        else:
            value = torch.tensor(value, dtype=torch.float32)

        W = value * torch.ones(self.d_out, self.d_in, dtype=value.dtype)

        return W
