import torch
import torch.nn as nn


class ComplexToReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.real
        x = self.activation(x)
        return x


class ComplexToRealGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
        self.glu = nn.GLU(dim=-2)

    def forward(self, x):
        x = torch.cat((x.real, x.imag), dim=-2)
        x = self.activation(x)
        x = self.glu(x)

        return x


class ComplexToRealABS(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = torch.abs(x)
        x = self.activation(x)

        return x


class ComplexToRealAngle(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = torch.angle(x)
        x = self.activation(x)

        return x


# class IQToReal(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     @staticmethod
#     def resample_poly_torch(signal, up, down):
#         # Convert the complex signal to separate real and imaginary parts
#         signal_real = torch.real(signal)
#         signal_imag = torch.imag(signal)
#
#         # Perform interpolation directly on the (B, H, L) shape tensors
#         signal_real_interp = nn.functional.interpolate(signal_real, scale_factor=up / down,
#                                                        mode='linear', align_corners=False)
#         signal_imag_interp = nn.functional.interpolate(signal_imag, scale_factor=up / down,
#                                                        mode='linear', align_corners=False)
#
#         # Downsample back to the original length
#         signal_real_downsampled = nn.functional.interpolate(signal_real_interp, size=signal_real.shape[-1],
#                                                             mode='linear', align_corners=False)
#         signal_imag_downsampled = nn.functional.interpolate(signal_imag_interp, size=signal_imag.shape[-1],
#                                                             mode='linear', align_corners=False)
#
#         # Combine real and imaginary parts back into a complex tensor
#         signal_interp = signal_real_downsampled + 1j * signal_imag_downsampled
#         return signal_interp
#
#     def forward(self, signal_iq):
#         fs_iq = 0.07  # Adjust this as necessary for your actual sampling frequency
#         signal_iq_interp = IQToReal.resample_poly_torch(signal_iq, up=2, down=1)
#
#         # Step 2: Shift IQ signal to positive frequencies
#         freq_shift = fs_iq / 2
#         fs_real = fs_iq * 2
#         time_vector = torch.arange(signal_iq_interp.shape[-1], device=signal_iq_interp.device)
#         complex_sine = torch.exp(1j * 2 * torch.pi * (freq_shift / fs_real) * time_vector)
#         signal_shifted = signal_iq_interp * complex_sine
#
#         # Step 3: Take real part
#         signal_real = torch.real(signal_shifted)
#
#         return signal_real
