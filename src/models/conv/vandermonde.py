import torch
from src.reservoir.state_reservoir import DiscreteStateReservoir, ContinuousStateReservoir
import torch.nn as nn


class VandermondeConv(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size, strong_stability, weak_stability, dt=None,
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

        super(VandermondeConv, self).__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input

        # Initialize parameters C and D
        C = torch.randn(self.d_output, self.d_state, dtype=torch.complex64)
        self.C = nn.Parameter(torch.view_as_real(C), requires_grad=True)  # (H, P, 2)
        self.D = nn.Parameter(torch.randn(self.d_output, self.d_input, dtype=torch.float32),
                              requires_grad=True)  # (H, H)

        B = torch.randn(self.d_state, self.d_input, dtype=torch.complex64)
        if dt is None:
            dr = DiscreteStateReservoir(self.d_state, strong_stability, weak_stability, field)
            Lambda_bar = dr.diagonal_state_matrix()
            B_bar = B
        elif dt > 0:
            cr = ContinuousStateReservoir(self.d_state, strong_stability, weak_stability, field)
            Lambda = cr.diagonal_state_matrix()
            Lambda_bar, B_bar = self._zoh(Lambda, B, dt)  # Discretization
        else:
            raise ValueError("Delta time dt must be positive: set dt>0 or None for 'discrete dynamics'.")

        # Initialize parameters Lambda_bar and B_bar
        self.Lambda_bar = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=True)  # (P, 2)
        self.B_bar = nn.Parameter(torch.view_as_real(B_bar), requires_grad=True)  # (P, H, 2)

        self.powers = nn.Parameter(torch.arange(kernel_size, dtype=torch.float32), requires_grad=False)

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
        :param Delta: Timestep (1 for each input dimension)
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
        vandermonde = Lambda_bar.unsqueeze(1) ** self.powers  # (P, L)
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

    def forward(self, u):
        """
        Apply the convolution to the input sequence
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        B_bar = torch.view_as_complex(self.B_bar)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)

        # Vandermonde matrix for kernel computation
        vandermonde = self._construct_vandermonde()  # (P, L)

        kernel = torch.einsum('hp,pl->hl', torch.einsum('hp,ph->hp', C, B_bar), vandermonde)  # (H, L)
        kernel_s = torch.fft.fft(kernel, dim=-1)

        y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, kernel_s), dim=-1)  # (B, H, L)

        y = y.real + torch.einsum('hh,bhl->bhl', self.D, u)  # (B, H, L)

        y = self.activation(y)

        return y, None


class VandermondeReservoirConv(VandermondeConv):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size, strong_stability, weak_stability, dt=None,
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
        # TODO: Hyperparameter dt>0 for continuous dynamics:
        #   Lambda_bar = Lambda_Bar(Lambda, dt), B_bar = B(Lambda, B, dt)

        super().__init__(d_input, d_state, kernel_size, strong_stability, weak_stability, dt, field)

        # Freeze Lambda_bar parameter
        self.Lambda_bar.requires_grad_(False)
        # Frozen Vandermonde matrix for kernel computation
        self.vandermonde = nn.Parameter(self._construct_vandermonde(), requires_grad=False)  # (P, L)

    def forward(self, u):
        """
        Apply the convolution to the input sequence
        :param u: batched input sequence of shape (B,H,L) = (batch_size, d_input, input_length)
        :return: y: batched output sequence of shape (B,H,L) = (batch_size, d_output, input_length)
        """
        B_bar = torch.view_as_complex(self.B_bar)  # (P, H)
        C = torch.view_as_complex(self.C)  # (H, P)

        u_s = torch.fft.fft(u, dim=-1)  # (B, H, L)

        kernel = torch.einsum('hp,pl->hl', torch.einsum('hp,ph->hp', C, B_bar), self.vandermonde)  # (H, L)
        kernel_s = torch.fft.fft(kernel, dim=-1)

        y = torch.fft.ifft(torch.einsum('bhl,hl->bhl', u_s, kernel_s), dim=-1)  # (B, H, L)

        y = y.real + torch.einsum('hh,bhl->bhl', self.D, u)  # (B, H, L)

        y = self.activation(y)

        return y, None
