import torch


def hilbert_transform(x):
    """
    Apply the Hilbert transform to a batched time series.

    Args:
    x (torch.Tensor): Input tensor of shape (B, H, L)

    Returns:
    tuple: A tuple containing:
        - torch.Tensor: Hilbert transformed signal of shape (B, H, L) with complex values
        - torch.Tensor: Frequencies after FFT of shape (B, H, L) with complex values
    """
    B, H, L = x.shape
    Xf = torch.fft.fft(x, dim=-1)  # FFT along the last dimension

    # Create the Hilbert transform filter
    h = torch.zeros(L, dtype=torch.complex64, device=x.device)
    if L % 2 == 0:
        h[0] = 1
        h[1:L // 2] = 2
        h[L // 2] = 1
    else:
        h[0] = 1
        h[1:(L + 1) // 2] = 2

    h = h.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, L) to broadcast over (B, H, L)

    # Apply the filter in the frequency domain
    Xf_filtered = Xf * h

    # Inverse FFT to get the Hilbert transform
    x_hilbert = torch.fft.ifft(Xf_filtered, dim=-1)

    return x_hilbert, Xf_filtered
