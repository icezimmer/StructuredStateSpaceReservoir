import torch
from sklearn.decomposition import PCA


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


def pca_analysis(tensor, n_components=1):
    # Ensure the tensor is 2D for PCA
    if tensor.dim() != 2:
        raise ValueError("Tensor must be 2-dimensional for PCA.")

    # Convert tensor to numpy array
    tensor_np = tensor.cpu().numpy()  # (num_samples, num_features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(tensor_np)

    # Loadings (coefficients of the principal components)
    loadings = pca.components_

    return loadings


def total_entropy(tensor, dim, bins=2):

    # Step 1: Flatten the tensor
    flat_tensor = tensor.flatten().contiguous()

    bins = bins ** dim  # Curse of dimensionality

    # Step 2: Discretize the values into bins
    min_value, max_value = flat_tensor.min(), flat_tensor.max()
    bin_edges = torch.linspace(min_value, max_value, steps=bins + 1).to(flat_tensor.device)
    bin_indices = torch.bucketize(flat_tensor, bin_edges, right=True)

    # Step 3: Compute the histogram of binned values
    unique_bins, counts = torch.unique(bin_indices, return_counts=True)

    # Step 4: Normalize the counts to form a probability distribution
    prob_dist = counts.float() / counts.sum()

    # Step 5: Compute the entropy
    entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9))  # Adding a small value to avoid log(0)

    return entropy
