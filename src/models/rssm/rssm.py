import torch
from src.models.rssm.convolutions.fft_reservoir import FFTConvReservoir


class RSSM(torch.nn.Module):
    def __init__(self, d_model, d_state,
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

        realfuns = ['real', 'real+relu', 'real+tanh']
        if fun_forward not in realfuns or fun_fit not in realfuns:
            raise ValueError('Real Function of Complex Vars must be one of {}'.format(realfuns))

        super().__init__()

        self.d_model = d_model  # d_input = d_output = d_model = H
        self.d_state = d_state  # d_state = P

        self.layer = FFTConvReservoir(d_input=self.d_model, d_state=self.d_state, kernel=kernel,
                                      **layer_args)

        if fun_forward == 'real':
            self.fun_forward = torch.nn.Identity()
        elif fun_forward == 'real+relu':
            self.fun_forward = torch.nn.ReLU()
        elif fun_forward == 'real+tanh':
            self.fun_forward = torch.nn.Tanh()

        if fun_fit == 'real':
            self.fun_fit = torch.nn.Identity()
        elif fun_fit == 'real+relu':
            self.fun_fit = torch.nn.ReLU()
        elif fun_fit == 'real+tanh':
            self.fun_fit = torch.nn.Tanh()

    def step(self, u, x):
        """
        Step one time step as a recurrent model.
        :param u: input step of shape (B, H)
        :param x: previous state of shape (B, P)
        :return: output step (B, H), new state (B, P)
        """
        u, x = self.layer.step(u, x)

        return self.fun_forward(u), self.fun_fit(u), x

    def forward(self, u):
        """
        Forward method for the RSSM layer.
        :param u: batched input sequence of shape (B, H, L) = (batch_size, d_input, input_length)
        :return:
            y: batched output sequence of shape (B, H, L) = (batch_size, d_output, input_length)
                as input of the next layer
            z: batched output sequence of shape (B, H, L) = (batch_size, d_output, input_length)
                to collect and train the readout
        """
        u, _ = self.layer(u)

        return self.fun_forward(u), self.fun_fit(u)
