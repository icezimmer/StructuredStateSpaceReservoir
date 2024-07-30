import torch
from src.models.rssm.convolutions.fft_reservoir import FFTConvReservoir
from src.models.rssm.realfun_complexvar.complex_to_real import Real, RealReLU, RealTanh, RealImagTanhGLU, ABSTanh, AngleTanh

"""
see: https://github.com/i404788/s5-pytorch/tree/74e2fdae00b915a62c914bf3615c0b8a4279eb84
"""


# TODO: replace mixing layer with only a non-linearity (identity is the best choice for mixing layer)
class RSSM(torch.nn.Module):
    def __init__(self, d_model,
                 realfun,
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
        if realfun not in realfuns:
            raise ValueError('Real Function of Complex Vars must be one of {}'.format(realfuns))

        super().__init__()

        self.d_model = d_model

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_model, kernel=kernel,
                                      **layer_args)

        self.postprocess = RealReLU()

        if realfun == 'real':
            self.realfun = Real()
        elif realfun == 'real+relu':
            self.realfun = RealReLU()
        elif realfun == 'real+tanh':
            self.realfun = RealTanh()
        elif realfun == 'glu':
            self.realfun = RealImagTanhGLU()
        elif realfun == 'abs+tanh':
            self.realfun = ABSTanh()
        elif realfun == 'angle+tanh':
            self.realfun = AngleTanh()

    # TODO: implement step method for mixing layer
    def step(self, u, x):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: output step (B, H), new state (B, P)
        """
        y, x = self.layer.step(u, x)
        y = self.postprocess(y)

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

        y = self.postprocess(u)
        z = self.realfun(u)

        return y, z
