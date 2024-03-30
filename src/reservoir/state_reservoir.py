import torch
import warnings


class DiscreteStateReservoir:
    def __init__(self, d_state, strong_stability, weak_stability, field):
        """
        Generate the discrete state matrix A_bar for an SSM model.
        Args:
            d_state:
            strong_stability:
            weak_stability:
            field:
        """
        self.d_state = d_state

        if strong_stability > weak_stability or strong_stability < 0:
            raise ValueError("For the discrete dynamics we must have:"
                             "0 <= strong_stability <= |lambda| < weak_stability.")

        if weak_stability > 1:
            warnings.warn("For a stable discrete dynamics set weak_stability such that:"
                          "0 <= strong_stability <= |lambda| < weak_stability <= 1.")

        self.strong_stability = strong_stability
        self.weak_stability = weak_stability

        if field not in ['complex', 'real']:
            raise ValueError("The field must be 'complex' or 'real'.")
        else:
            self.field = field

    def diagonal_state_matrix(self):
        """
        Create a state matrix Lambda_bar for the discrete dynamics;
        lambda = radius * (cos(theta) + i * sin(theta)):
        radius in [strong_stability, weak_stability),
        theta in [0, 2pi).
        :return: Lambda_bar
        """
        if self.field == 'complex':
            radius = (self.strong_stability + (self.weak_stability - self.strong_stability) *
                      torch.rand(self.d_state, dtype=torch.float32))
            theta = 2 * torch.pi * torch.rand(self.d_state, dtype=torch.float32)
            alpha_tensor = radius * torch.cos(theta)
            omega_tensor = radius * torch.sin(theta)
        elif self.field == 'real':
            half_d_state = self.d_state // 2
            radius = (self.strong_stability + (self.weak_stability - self.strong_stability) *
                      torch.rand(half_d_state, dtype=torch.float32))
            theta = torch.pi * torch.rand(half_d_state, dtype=torch.float32)
            alpha_tensor = torch.cat((radius * torch.cos(theta), radius * torch.cos(theta)), 0)
            omega_tensor = torch.cat((radius * torch.sin(theta), -radius * torch.sin(theta)), 0)
            if self.d_state % 2 == 1:
                extra_radius = (self.strong_stability + (self.weak_stability - self.strong_stability) *
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
    def __init__(self, d_state, strong_stability, weak_stability, field):
        """
        Generate the continuous state matrix A for an SSM model.
        :param d_state:
        :param strong_stability:
        :param weak_stability:
        :param field:
        """
        self.d_state = d_state

        if strong_stability > weak_stability:
            raise ValueError("For the continuous dynamics we must have:"
                             "strong_stability <= Re(lambda) < weak_stability.")

        if weak_stability > 0:
            warnings.warn("For a stable continuous dynamics set weak_stability such that:"
                          "strong_stability <= Re(lambda) < weak_stability <= 0.")

        self.strong_stability = strong_stability
        self.weak_stability = weak_stability

        if field not in ['complex', 'real']:
            raise ValueError("The field must be 'complex' or 'real'.")
        else:
            self.field = field

    def diagonal_state_matrix(self):
        """
        Create a state matrix Lambda for the continuous dynamics;
        lambda = log(radius) + i * theta:
        Re(lambda) in [min_real, max_real) = [log(min_radius), log(max_radius)),
        Im(lambda) in [0, 2pi).
        :return: Lambda
        """
        min_real = self.strong_stability
        max_real = self.weak_stability
        if self.field == 'complex':
            real_tensor = min_real + (max_real - min_real) * torch.rand(self.d_state, dtype=torch.float32)
            imag_tensor = 2 * torch.pi * torch.rand(self.d_state, dtype=torch.float32)
        elif self.field == 'real':
            half_d_state = self.d_state // 2
            real_tensor = min_real + (max_real - min_real) * torch.rand(half_d_state, dtype=torch.float32)
            imag_tensor = torch.pi * torch.rand(half_d_state, dtype=torch.float32)
            real_tensor = torch.cat((real_tensor, real_tensor), 0)
            imag_tensor = torch.cat((imag_tensor, -imag_tensor), 0)
            if self.d_state % 2 == 1:
                extra_real = min_real + (max_real - min_real) * torch.rand(1, dtype=torch.float32)
                # Choose 0 or pi randomly for extra_imag (extra_theta)
                extra_imag = torch.randint(0, 2, (1,)) * torch.pi
                real_tensor = torch.cat((real_tensor, extra_real), 0)
                imag_tensor = torch.cat((imag_tensor, extra_imag), 0)
        else:
            raise ValueError("The field must be 'complex' or 'real'.")

        Lambda = torch.complex(real_tensor, imag_tensor)
        return Lambda
