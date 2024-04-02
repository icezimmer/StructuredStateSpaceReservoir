def standardize(data):
    """
    Standardize the data by subtracting the mean and dividing by the standard deviation.
    args:
        data: torch.Tensor of shape (?, ..., ?, L) = (?, ..., ?, length)
    returns:
        data: (?, ..., ?, L)
        mean: (?, ..., ?, 1)
        std: (?, ..., ?, 1)
    """
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, keepdim=True)
    return (data - mean) / std, mean, std

