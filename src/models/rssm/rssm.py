import torch
from src.models.rssm.convolutions.fft_reservoir import FFTConvReservoir
from src.models.rssm.realfun_complexvar.complex_to_real import Real, RealReLU, RealTanh, RealImagTanhGLU, ABSTanh, AngleTanh

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


# TODO: replace mixing layer with only a non-linearity (identity is the best choice for mixing layer)
class RSSM(torch.nn.Module):
    def __init__(self, d_model,
                 fun_forward,
                 fun_fit,
                 kernel,
                 **layer_args):
        """
        RSSM model.
        :param d_input: dimensionality of the input space
        :param d_state: dimensionality of the latent space
        :param dt: delta time for continuous dynamics (default: None for discrete dynamics)
        :param field: field for the state 'real' or 'complex' (default: 'complex')
        """

        kernel_classes = ['Vr']
        if kernel not in kernel_classes:
            raise ValueError('Kernel must be one of {}'.format(kernel_classes))

        realfuns = ['real', 'real+relu', 'real+tanh', 'glu', 'abs+tanh', 'angle+tanh']
        if fun_forward not in realfuns or fun_fit not in realfuns:
            raise ValueError('Real Function of Complex Vars must be one of {}'.format(realfuns))

        super().__init__()

        self.d_model = d_model

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model, kernel=kernel,
                                      **layer_args)

        if fun_forward == 'real':
            self.fun_forward = Real()
        elif fun_forward == 'real+relu':
            self.fun_forward = RealReLU()
        elif fun_forward == 'real+tanh':
            self.fun_forward = RealTanh()
        elif fun_forward == 'glu':
            self.fun_forward = RealImagTanhGLU()
        elif fun_forward == 'abs+tanh':
            self.fun_forward = ABSTanh()
        elif fun_forward == 'angle+tanh':
            self.fun_forward = AngleTanh()

        if fun_fit == 'real':
            self.fun_fit = Real()
        elif fun_fit == 'real+relu':
            self.fun_fit = RealReLU()
        elif fun_fit == 'real+tanh':
            self.fun_fit = RealTanh()
        elif fun_fit == 'glu':
            self.fun_fit = RealImagTanhGLU()
        elif fun_fit == 'abs+tanh':
            self.fun_fit = ABSTanh()
        elif fun_fit == 'angle+tanh':
            self.fun_fit = AngleTanh()

    # TODO: implement step method for mixing layer
    def step(self, u, x):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: output step (B, H), new state (B, P)
        """
        y, x = self.layer.step(u, x)
        y = self.fun_forward(y)

        return y, x

    def forward(self, u):
        """
        Forward method for the RSSM layer.
        :param u: batched input sequence of shape (B, H, L) = (batch_size, d_input, input_length)
        :return:
            y: batched output sequence of shape (B, H, L) = (batch_size, d_output, input_length)
                as inout of the next layer
            z: batched output sequence of shape (B, H, L) = (batch_size, d_output, input_length)
                to collect and train the readout
        """
        u, _ = self.layer(u)

        y = self.fun_forward(u)
        z = self.fun_fit(u)

        return y, z
