import torch
import warnings


class DiscreteStateReservoir:
    def __init__(self, d_state):
        """
        Generate the discrete state matrix.
        Args:
            d_state: int, the dimension of the state space
        """
        self.d_state = d_state

    def echo_state_matrix(self, max_radius):
        if max_radius > 1:
            warnings.warn("For a stable discrete dynamics set max_radius such that:"
                          "0 <= min_radius <= |lambda| < max_radius <= 1.")

        w_hh = -1 + 2 * torch.rand(self.d_state, self.d_state, dtype=torch.float32)
        eigenvalues = torch.linalg.eigvals(w_hh)
        w_hh = (max_radius / torch.max(torch.abs(eigenvalues))) * w_hh
        return w_hh

    def diagonal_state_space_matrix(self, min_radius, max_radius, field):
        """
        Create a state matrix Lambda_bar for the discrete dynamics;
        lambda = radius * (cos(theta) + i * sin(theta)):
        radius in [min_radius, max_radius),
        theta in [0, 2pi).
        :return: Lambda_bar
        """
        if min_radius > max_radius or min_radius < 0:
            raise ValueError("For the discrete dynamics we must have:"
                             "0 <= min_radius <= |lambda| < max_radius.")
        if max_radius > 1:
            warnings.warn("For a stable discrete dynamics set max_radius such that:"
                          "0 <= min_radius <= |lambda| < max_radius <= 1.")

        if field == 'complex':
            radius = (min_radius + (max_radius - min_radius) *
                      torch.sqrt(torch.rand(self.d_state, dtype=torch.float32)))
            theta = 2 * torch.pi * torch.rand(self.d_state, dtype=torch.float32)
            alpha_tensor = radius * torch.cos(theta)
            omega_tensor = radius * torch.sin(theta)
        elif field == 'real':
            half_d_state = self.d_state // 2
            radius = (min_radius + (max_radius - min_radius) *
                      torch.sqrt(torch.rand(half_d_state, dtype=torch.float32)))
            theta = torch.pi * torch.rand(half_d_state, dtype=torch.float32)
            alpha_tensor = torch.cat((radius * torch.cos(theta), radius * torch.cos(theta)), 0)
            omega_tensor = torch.cat((radius * torch.sin(theta), -radius * torch.sin(theta)), 0)
            if self.d_state % 2 == 1:
                extra_radius = (min_radius + (max_radius - min_radius) *
                                torch.rand(1, dtype=torch.float32))
                # Choose 0 or pi randomly for extra_theta
                extra_theta = torch.randint(0, 2, (1,)) * torch.pi
                alpha_tensor = torch.cat((alpha_tensor, extra_radius * torch.cos(extra_theta)), 0)
                omega_tensor = torch.cat((omega_tensor, extra_radius * torch.sin(extra_theta)), 0)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        Lambda_bar = torch.complex(alpha_tensor, omega_tensor)
        return Lambda_bar


class ContinuousStateReservoir:
    def __init__(self, d_state):
        """
        Generate the continuous state matrix.
        :param d_state: int, the dimension of the state space
        """
        self.d_state = d_state

    def echo_state_matrix(self, max_real_part):
        if max_real_part > 0:
            warnings.warn("For a stable continuous dynamics set max_real_part such that:"
                          "Re(lambda) < max_real_part <= 0.")

        real_parts = max_real_part - (1 + torch.abs(max_real_part)) * torch.rand(self.d_state, dtype=torch.float32)
        imag_parts = 2 * torch.pi * torch.rand(self.d_state, dtype=torch.float32)
        eigenvalues = torch.complex(real_parts, imag_parts)

        D = torch.diag(eigenvalues)
        Q, _ = torch.linalg.qr(torch.randn(self.d_state, self.d_state, dtype=torch.complex64))
        w_hh = Q @ D @ Q.T.conj()

        return w_hh

    def diagonal_state_space_matrix(self, min_real_part, max_real_part, field):
        """
        Create a state matrix Lambda for the continuous dynamics;
        lambda = log(radius) + i * theta:
        Re(lambda) in [min_real_part, max_real_part) = [log(min_radius), log(max_radius)),
        Im(lambda) in [0, 2pi).
        :return: Lambda
        """
        if min_real_part > max_real_part:
            raise ValueError("For the continuous dynamics we must have:"
                             "min_real_part <= Re(lambda) < max_real_part.")
        if max_real_part > 0:
            warnings.warn("For a stable continuous dynamics set max_real_part such that:"
                          "min_real_part <= Re(lambda) < max_real_part <= 0.")

        if field == 'complex':
            real_tensor = (min_real_part +
                           (max_real_part - min_real_part) * torch.rand(self.d_state, dtype=torch.float32))
            imag_tensor = 2 * torch.pi * torch.rand(self.d_state, dtype=torch.float32)
        elif field == 'real':
            half_d_state = self.d_state // 2
            real_tensor = (min_real_part +
                           (max_real_part - min_real_part) * torch.rand(half_d_state, dtype=torch.float32))
            imag_tensor = torch.pi * torch.rand(half_d_state, dtype=torch.float32)
            real_tensor = torch.cat((real_tensor, real_tensor), 0)
            imag_tensor = torch.cat((imag_tensor, -imag_tensor), 0)
            if self.d_state % 2 == 1:
                extra_real = min_real_part + (max_real_part - min_real_part) * torch.rand(1, dtype=torch.float32)
                # Choose 0 or pi randomly for extra_imag (extra_theta)
                extra_imag = torch.randint(0, 2, (1,)) * torch.pi
                real_tensor = torch.cat((real_tensor, extra_real), 0)
                imag_tensor = torch.cat((imag_tensor, extra_imag), 0)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        Lambda = torch.complex(real_tensor, imag_tensor)
        return Lambda
