import torch


def get_fft(y: torch.Tensor, dim=None):
    """Calculate DFT of y.

    Inputs:
    - y: Tensor of shape (C, ...)
    - dim: dimensions to calculate DFT over.

    Outputs:
    - y_fft: Absolute values of DFT of y, summed across C.
    """
    if y.ndim == 2:
        y = y[None, :, :]  # add axis for channels
    if dim is None:
        dim = tuple(range(1, y.ndim))

    if y.ndim not in {3, 4}:
        raise ValueError("y should have 3 or 4 dimensions.")

    y_fft = torch.fft.fftn(y, dim=dim)
    y_fft = torch.abs(y_fft)
    return y_fft.sum(dim=0)  # sum across channels


def _spectrum_from_fft(y: torch.Tensor):
    if y.ndim == 2:
        mask = get_2d_fft_mask(y.shape[0]).to(device=y.device)
        y = torch.flip(torch.abs(y) * mask, dims=[0])
        return torch.tensor(
            [
                torch.sum(torch.diag(y, diagonal=k))
                for k in range(-y.shape[0] + 1, y.shape[0])
            ]
        ).to(device=y.device)
    elif y.ndim == 3:
        mask = get_3d_fft_mask(y.shape[0]).to(device=y.device)
        y = torch.flip(torch.abs(y) * mask, dims=[1])
        result = torch.zeros(y.shape[0] * 3 - 2).to(device=y.device)
        y_slice: torch.Tensor
        for i, y_slice in enumerate(y):
            result[i : i + y.shape[0] * 2 - 1] += torch.tensor(
                [
                    torch.sum(torch.diag(y_slice, diagonal=k))
                    for k in range(-y_slice.shape[0] + 1, y_slice.shape[0])
                ]
            ).to(device=y.device)
        return result


def get_2d_fft_mask(sidelength):
    """Mask not-redundant frequencies in 3D FFT."""
    n_unique = sidelength // 2  # number of unique frequencies on diagonal
    mask = torch.triu(torch.ones(sidelength, sidelength), diagonal=0)
    mask += torch.diag(
        torch.concatenate(
            [torch.ones(n_unique), torch.zeros(sidelength - 1 - n_unique)], dim=0
        ),
        diagonal=-1,
    )
    mask = mask.flip(1)

    mask[1 + n_unique :, 0] = 0
    mask[0, 1 + n_unique :] = 0
    return mask


def get_3d_fft_mask(sidelength):
    """Mask not-redundant frequencies in 3D FFT."""
    m = torch.zeros((sidelength, sidelength, sidelength))
    n = (sidelength + 1) // 2

    mask_2d = get_2d_fft_mask(sidelength)
    m[0] = mask_2d
    m[:, 0] = mask_2d
    m[:, :, 0] = mask_2d

    for i in range(1, sidelength):
        d = sidelength // 2 - i
        if i < sidelength // 2 or sidelength % 2 == 1:
            d += 1

        t = 1 - torch.triu(torch.ones(sidelength - 1, sidelength - 1), diagonal=d).flip(
            0
        )
        if i == sidelength / 2:  # applies only when sidelength is even
            t = t + torch.diag(
                torch.concatenate(
                    [torch.zeros(n - 1), torch.ones(sidelength - n)], dim=0
                ),
                diagonal=0,
            ).flip(0)
        m[i, 1:, 1:] = t
    return m


def get_spectrum(y: torch.Tensor, drop_bias=True):
    spectrum = _spectrum_from_fft(get_fft(y))
    if drop_bias:  # bias can be easily modelled in the last layer, so it's unnecessary
        return spectrum[1:]
    else:
        return spectrum
