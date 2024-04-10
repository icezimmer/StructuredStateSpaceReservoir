import torch
from src.reservoir.state_reservoir import DiscreteStateReservoir, ContinuousStateReservoir
from src.reservoir.reservoir import Reservoir
import torch.nn as nn


class VandermondeKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability, dt=None,
                 drop_kernel=0.0, dropout=0.0,
                 field='complex'):
        """
        Construct an SSM model with frozen state matrix Lambda_bar:
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        # TODO: Delta trainable parameter not fixed to ones for continuous dynamics:
        #   Lambda_bar = Lambda_Bar(Lambda, Delta), B_bar = B(Lambda, B, Delta)

        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # Necessary condition for the Vandermonde kernel (SISO)

        input2state_reservoir = Reservoir(d_in=self.d_input, d_out=self.d_state)
        state2output_reservoir = Reservoir(d_in=self.d_state, d_out=self.d_output)

        B = input2state_reservoir.uniform_matrix(scaling=1.0, field=field)
        C = state2output_reservoir.uniform_matrix(scaling=1.0, field=field)

        if dt is None:
            state_reservoir = DiscreteStateReservoir(self.d_state)
            Lambda_bar = state_reservoir.diagonal_state_space_matrix(
                min_radius=strong_stability, max_radius=weak_stability, field=field)
            B_bar = B
        elif dt > 0:
            state_reservoir = ContinuousStateReservoir(self.d_state)
            Lambda = state_reservoir.diagonal_state_space_matrix(
                min_real_part=strong_stability, max_real_part=weak_stability, field=field)
            Lambda_bar, B_bar = self._zoh(Lambda, B, dt)
        else:
            raise ValueError("Delta time dt must be positive: set dt>0 or None for 'discrete dynamics'.")

        # Initialize parameters Lambda_bar and B_bar
        self.Lambda_bar = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=True)  # (P, 2)
        self.B_bar = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, H, 2)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (H, P, 2)

        self.powers = torch.arange(kernel_size, dtype=torch.float32)

        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.activation = nn.Tanh()


    @staticmethod
    def _zoh(Lambda, B, dt):
        """
        Discretize the system using the zero-order-hold transform:
        Lambda_bar = exp(Lambda*Delta)
        B_bar = (A_bar - I) * inv(A) * B
        C = C
        D = D
        :param Lambda: State Diagonal Matrix (Continuous System)
        :param B: Input->State Matrix (Continuous System)
        :param dt: Delta time for discretization
        :return: Lambda_bar, B_bar (Discrete System)
        """
        Ones = torch.ones(Lambda.shape[0], dtype=torch.float32)

        # Lambda_bar = torch.exp(torch.mul(Lambda, Delta))
        Lambda_bar = torch.exp(Lambda * dt)

        B_bar = torch.einsum('p,ph->ph', torch.mul(1 / Lambda, (Lambda_bar - Ones)), B)

        return Lambda_bar, B_bar

    def _construct_vandermonde(self):
        """
        Args:
            kernel_size:

        Returns:

        """
        Lambda_bar = torch.view_as_complex(self.Lambda_bar)
        powers = self.powers.to(device=Lambda_bar.device)
        vandermonde = Lambda_bar.unsqueeze(1) ** powers  # (P, L)
        return vandermonde

    def step(self, u, x):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
        x: (B, H)
        state: (B, P)
        Returns: y (B, H), state (B, P)
        """
        u = torch.view_as_complex(u)
        A_bar = torch.view_as_complex(self.Lambda_bar)  # (P)
        B_bar = torch.view_as_complex(self.B_bar)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)

        x = torch.einsum('p,bp->bp', A_bar, x) + torch.einsum('ph,bh->bp', B_bar, u)  # (B,P)
        y = torch.einsum('hp,bp->bh', C, x).real + torch.einsum('hh,bh->bh', self.D, u)  # (B,H)
        y = self.activation(y)

        return y, x

    def forward(self):
        """
        Apply the convolution to the input sequence
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        vandermonde = self._construct_vandermonde()  # (P, L)
        B_bar = torch.view_as_complex(self.B_bar)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)

        kernel = torch.einsum('hp,pl->hl', torch.einsum('hp,ph->hp', C, B_bar), vandermonde)  # (H, L)

        kernel = self.drop_kernel(kernel)

        return kernel, None


class VandermondeStateReservoirKernel(VandermondeKernel):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability, dt=None,
                 drop_kernel=0.0, dropout=0.0,
                 field='complex'):
        """
        Construct an SSM model with frozen state matrix Lambda_bar:
        x_new = Lambda_bar * x_old + B_bar * u_new
        y_new = C * x_new + D * u_new
        :param
            d_input: dimensionality of the input space
            d_state: dimensionality of the latent space
            kernel_size: size of the convolution kernel (length of the input sequence)
            dt: delta time for continuous dynamics (default: None for discrete dynamics)
            field: field for the state 'real' or 'complex' (default: 'complex')
        """
        # TODO: Hyperparameter dt>0 for continuous dynamics:
        #   Lambda_bar = Lambda_Bar(Lambda, dt), B_bar = B(Lambda, B, dt)

        super().__init__(d_input, d_state, kernel_size,
                         strong_stability, weak_stability, dt,
                         drop_kernel, dropout,
                         field)

        # Freeze Lambda_bar parameter
        self.Lambda_bar.requires_grad_(False)
        # Frozen Vandermonde matrix for kernel computation
        self.vandermonde = nn.Parameter(self._construct_vandermonde(), requires_grad=False)  # (P, L)

    def forward(self):
        """
        Generate the convolution kernel from the diagonal SSM parameters
        """
        B_bar = torch.view_as_complex(self.B_bar)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)
        kernel = torch.einsum('hp,pl->hl', torch.einsum('hp,ph->hp', C, B_bar), self.vandermonde)  # (H, L)

        kernel = self.drop_kernel(kernel)

        return kernel, None
