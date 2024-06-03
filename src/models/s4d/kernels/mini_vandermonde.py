import torch
from src.reservoir.state import DiscreteStateReservoir
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn
import warnings


class MiniVandermonde(nn.Module):
    """
    Generate convolution kernel from diagonal SSM parameters
    """

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability,
                 min_scaleW=0.0, max_scaleW=1.0,
                 lr=0.001, wd=0.0,
                 field='complex'):
        """
        Construct the convolution kernel.
        Assuming diagonal state matrix A of shape (d_state), the Mini-Vandermonde Kernel is:
            kernel[i,l] = W[i,:] * diag(A)^l = (sqrt(W[i,:]) .* sqrt(W^t[:,i])) * A^l
        where:
            diag(A)[j,j] = A[j]
            A^l[j] = A[j]^l
        and:
            j = 0, ..., d_state-1
            i = 0, ..., d_input-1
            l = 0, ..., kernel_size-1.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """

        super().__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = self.d_input  # Necessary condition for the Vandermonde kernel (SISO)

        self.register_buffer('x0', torch.zeros(self.d_state, dtype=torch.complex64))

        input_output_reservoir = ReservoirMatrix(d_in=self.d_state, d_out=self.d_output)

        W = input_output_reservoir.uniform_ring(min_radius=min_scaleW, max_radius=max_scaleW, field=field)

        state_reservoir = DiscreteStateReservoir(self.d_state)
        Lambda_bar = state_reservoir.diagonal_state_space_matrix(
            min_radius=strong_stability, max_radius=weak_stability, field=field)

        self.A = nn.Parameter(torch.view_as_real(Lambda_bar), requires_grad=True)  # (P, 2)
        self.W = nn.Parameter(torch.view_as_real(W), requires_grad=True)  # (P, H, 2)

        if lr < 0.0 or wd < 0.0:
            raise ValueError("Learning rate an weight decay for kernel parameters bust be positive.")
        if lr > 0.001 or wd > 0.0:
            warnings.warn("For a better optimization of the kernel parameters set lr <= 0.001 and wd = 0.0.")

        self.A._optim = {'lr': lr, 'weight_decay': wd}
        self.W._optim = {'lr': lr, 'weight_decay': wd}

        # Register powers for Vandermonde matrix
        powers = torch.arange(kernel_size, dtype=torch.float32)
        self.register_buffer('powers', powers)  # (L)

    def _freeze_parameter(self, param_name):
        """
        Converts a parameter to a buffer, effectively freezing it.
        This means the parameter will no longer be updated during training.

        Args:
            param_name (str): The name of the parameter to freeze.
        """
        # Ensure the attribute exists and is a parameter
        if hasattr(self, param_name) and isinstance(getattr(self, param_name), nn.Parameter):
            # Convert to buffer
            param = getattr(self, param_name).data
            delattr(self, param_name)  # Remove as parameter
            self.register_buffer(param_name, param)  # Register as buffer
        else:
            raise ValueError(f"{param_name} is not a parameter in this module.")

    def _construct_vandermonde(self):
        """
        Construct the Vandermonde Matrix from the diagonal state matrix A:
            V[j,l] = A[j]^l,
        where:
            j = 0, ..., d_state-1
            l = 0, ..., kernel_size-1.
        returns: V: (P,L)
        """
        A = torch.view_as_complex(self.A)
        V = A.unsqueeze(-1) ** self.powers  # (P, L)
        return V

    def step(self, u, x=None):
        """
        Step one time step as a recurrent model. Intended to be used during validation.
            x_new = A * x_old + u_new
            y_new = W * x_new,
        that is equivalent for y to:
            x_new = A * x_old + sqrt(W^t) * u_new
            y_new = sqrt(W) * x_new,
        :param u: time step input of shape (B, H)
        :param x: time step state of shape (B, P)
        :return: y: time step output of shape (B, H), x: time step state of shape (B, P)
        """
        u = u.to(dtype=torch.complex64)
        A = torch.view_as_complex(self.A)  # (P)
        W = torch.view_as_complex(self.W)  # (H, P)
        B = torch.sqrt(torch.transpose(W, 0, 1))  # (P, H)
        C = torch.sqrt(W)  # (H, P)

        if x is None:
            x = self.x0.unsqueeze(0).expand(u.shape[0], -1)
        x = torch.einsum('p,bp->bp', A, x) + torch.einsum('ph,bh->bp', B, u)  # (B,P)
        y = torch.einsum('hp,bp->bh', C, x).real  # (B,H)

        return y, x

    def forward(self):
        """
        Apply the convolution to the input sequence
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        V = self._construct_vandermonde()  # (P, L)
        W = torch.view_as_complex(self.W)  # (H, P)

        kernel = torch.einsum('hp,pl->hl', W, V)  # (H, L)

        return kernel, None


class MiniVandermondeFreezeW(MiniVandermonde):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability,
                 min_scaleW=0.0, max_scaleW=1.0,
                 lr=0.001, wd=0.0,
                 field='complex'):
        """
        Construct the convolution kernel with frozen W.
        Assuming diagonal state matrix A of shape (d_state), the Mini-Vandermonde Kernel is:
            kernel[i,l] = W[i,:] * diag(A)^l = (sqrt(W[i,:]) .* sqrt(W^t[:,i])) * A^l
        where:
            diag(A)[j,j] = A[j]
            A^l[j] = A[j]^l
        and:
            j = 0, ..., d_state-1
            i = 0, ..., d_input-1
            l = 0, ..., kernel_size-1.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        super().__init__(d_input, d_state, kernel_size,
                         strong_stability, weak_stability,
                         min_scaleW, max_scaleW,
                         lr, wd,
                         field)

        self._freeze_parameter('W')


class MiniVandermondeFreezeA(MiniVandermonde):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_input, d_state, kernel_size,
                 strong_stability, weak_stability,
                 min_scaleW=0.0, max_scaleW=1.0,
                 lr=0.001, wd=0.0,
                 field='complex'):
        """
        Construct the convolution kernel with frozen A.
        Assuming diagonal state matrix A of shape (d_state), the Mini-Vandermonde Kernel is:
            kernel[i,l] = W[i,:] * diag(A)^l = (sqrt(W[i,:]) .* sqrt(W^t[:,i])) * A^l
        where:
            diag(A)[j,j] = A[j]
            A^l[j] = A[j]^l
        and:
            j = 0, ..., d_state-1
            i = 0, ..., d_input-1
            l = 0, ..., kernel_size-1.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """
        super().__init__(d_input, d_state, kernel_size,
                         strong_stability, weak_stability,
                         min_scaleW, max_scaleW,
                         lr, wd,
                         field)

        self._freeze_parameter('A')

        # Register the Vandermonde matrix as buffer
        V = self._construct_vandermonde()  # (P, L)
        self.register_buffer('V', V)

    def forward(self):
        """
        Generate the convolution kernel from the diagonal SSM parameters
        :return: kernel: 1d convolution kernel of shape (H, L)
        """
        W = torch.view_as_complex(self.W)  # (H, P)
        kernel = torch.einsum('hp,pl->hl', W, self.V)  # (H, L)

        return kernel, None