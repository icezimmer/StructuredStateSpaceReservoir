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

    def echo_state_matrix(self, max_radius, leakage_rate):
        if max_radius < 0:
            raise ValueError("Max radius must be non-negative")
        if max_radius > 1:
            warnings.warn("For a stable discrete dynamics set max_radius such that:"
                          "0 <= |lambda| < max_radius <= 1.")

        if leakage_rate < 0 or leakage_rate > 1:
            raise ValueError("Leakage rate must be in [0, 1].")
        if leakage_rate > 0:
            w_hh = -1 + 2 * torch.rand(self.d_state, self.d_state, dtype=torch.float32)
            w_tilde = (1 - leakage_rate) * torch.eye(self.d_state, dtype=torch.float32) + leakage_rate * w_hh
            eigenvalues = torch.linalg.eigvals(w_tilde)
            w_tilde = (max_radius / torch.max(torch.abs(eigenvalues))) * w_tilde
            w_hh = (1 / leakage_rate) * (w_tilde - (1 - leakage_rate) * torch.eye(self.d_state, dtype=torch.float32))
        else:
            w_hh = torch.zeros(self.d_state, self.d_state, dtype=torch.float32)
        return w_hh

    def diagonal_state_space_matrix(self, min_radius, max_radius, min_theta, max_theta, field, n_ssm=None):
        """
        Create a state matrix Lambda_bar for the discrete dynamics;
        lambda = radius * (cos(theta) + i * sin(theta)):
        radius in [min_radius, max_radius),
        theta in [0, 2pi).
        :return: Lambda_bar
        """
        if n_ssm is not None and n_ssm <= 0:
            raise ValueError("The input dimension must be a positive integer.")

        if min_radius > max_radius or min_radius < 0:
            raise ValueError("For the discrete dynamics we must have:"
                             "0 <= min_radius <= |lambda| < max_radius.")
        if max_radius > 1.0:
            warnings.warn("For a stable discrete dynamics set max_radius such that:"
                          "0 <= min_radius <= |lambda| < max_radius <= 1.")
        if max_radius > 1.0:
            warnings.warn("For a stable discrete dynamics set max_radius such that:"
                          "0 <= min_radius <= |lambda| < max_radius <= 1.")

        if min_theta > max_theta or min_theta < 0.0:
            raise ValueError("For the discrete dynamics we must have:"
                             "0 <= min_theta <= theta < max_theta.")
        if max_theta > 2 * torch.pi:
            warnings.warn("Being max_theta > 2pi, max_theta is clipped to 2pi")
            max_theta = 2 * torch.pi

        if field == 'complex':
            if n_ssm is None:
                size = (self.d_state,)
            else:
                size = (self.d_state, n_ssm)
            radius = (min_radius + (max_radius - min_radius) *
                      torch.sqrt(torch.rand(size=size, dtype=torch.float32)))
            theta = min_theta + (max_theta - min_theta) * torch.rand(size=size, dtype=torch.float32)
            alpha_tensor = radius * torch.cos(theta)
            omega_tensor = radius * torch.sin(theta)
        elif field == 'real':
            if n_ssm is None:
                size = (self.d_state // 2,)
            else:
                size = (self.d_state // 2, n_ssm)
            radius = (min_radius + (max_radius - min_radius) *
                      torch.sqrt(torch.rand(size=size, dtype=torch.float32)))
            max_theta = min(torch.pi, max_theta)
            theta = min_theta + (max_theta - min_theta) * torch.rand(size=size, dtype=torch.float32)
            alpha_tensor = torch.cat(tensors=(radius * torch.cos(theta), radius * torch.cos(theta)), dim=0)
            omega_tensor = torch.cat(tensors=(radius * torch.sin(theta), -radius * torch.sin(theta)), dim=0)
            if self.d_state % 2 == 1:
                if n_ssm is None:
                    size = (1,)
                else:
                    size = (1, n_ssm)
                extra_radius = (min_radius + (max_radius - min_radius) *
                                torch.rand(size=size, dtype=torch.float32))
                # Choose 0 or pi randomly for extra_theta
                extra_theta = torch.randint(0, 2, (1,)) * torch.pi
                alpha_tensor = torch.cat(tensors=(alpha_tensor, extra_radius * torch.cos(extra_theta)), dim=0)
                omega_tensor = torch.cat(tensors=(omega_tensor, extra_radius * torch.sin(extra_theta)), dim=0)
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

    def diagonal_state_space_matrix(self, min_real_part, max_real_part, field, n_ssm=None):
        """
        Create a state matrix Lambda for the continuous dynamics;
        lambda = log(radius) + i * theta:
        Re(lambda) in [min_real_part, max_real_part) = [log(min_radius), log(max_radius)),
        Im(lambda) in [0, 2pi).
        :return: Lambda
        """
        if n_ssm is not None and n_ssm <= 0:
            raise ValueError("The input dimension must be a positive integer.")

        if min_real_part > max_real_part:
            raise ValueError("For the continuous dynamics we must have:"
                             "min_real_part <= Re(lambda) < max_real_part.")
        if max_real_part > 0.0:
            warnings.warn("For a stable continuous dynamics set max_real_part such that:"
                          "min_real_part <= Re(lambda) < max_real_part <= 0.")

        if field == 'complex':
            if n_ssm is None:
                size = (self.d_state,)
            else:
                size = (self.d_state, n_ssm)
            real_tensor = (min_real_part +
                           (max_real_part - min_real_part) * torch.rand(size=size, dtype=torch.float32))
            imag_tensor = 2 * torch.pi * torch.rand(size=size, dtype=torch.float32)
        elif field == 'real':
            if n_ssm is None:
                size = (self.d_state // 2,)
            else:
                size = (self.d_state // 2, n_ssm)
            real_tensor = (min_real_part +
                           (max_real_part - min_real_part) * torch.rand(size=size, dtype=torch.float32))
            imag_tensor = torch.pi * torch.rand(size=size, dtype=torch.float32)
            real_tensor = torch.cat((real_tensor, real_tensor), 0)
            imag_tensor = torch.cat((imag_tensor, -imag_tensor), 0)
            if self.d_state % 2 == 1:
                if n_ssm is None:
                    size = (1,)
                else:
                    size = (1, n_ssm)
                extra_real = (min_real_part + (max_real_part - min_real_part) *
                              torch.rand(size=size, dtype=torch.float32))
                # Choose 0 or pi randomly for extra_imag (extra_theta)
                extra_imag = torch.randint(0, 2, (1,)) * torch.pi
                real_tensor = torch.cat(tensors=(real_tensor, extra_real), dim=0)
                imag_tensor = torch.cat(tensors=(imag_tensor, extra_imag), dim=0)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        Lambda = torch.complex(real_tensor, imag_tensor)
        return Lambda
