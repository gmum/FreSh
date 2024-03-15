import sys
from typing import List

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from config_utils import SpectrumConfig
from spectum_utils import get_spectrum
from data_utils import resize_to_square


def _get_int(sizes: List[float | int]) -> List[int]:
    return list(filter(lambda x: isinstance(x, int), sizes))


def _get_float(sizes: List[float | int]) -> List[float]:
    return list(filter(lambda x: isinstance(x, float), sizes))


def _calculate_perc_sidelength(img, percentage):
    sidelength = torch.tensor(img.shape[1:]).min()
    return int(sidelength * percentage)


def prepare_spectrum_generators(config: SpectrumConfig):
    if np.any([p <= 0 or p > 1 for p in _get_float(config.resize_sizes)]):
        raise ValueError("Percentage sizes should be in (0, 1]")

    spectrum_generators = dict()
    if config.use_baseline:
        spectrum_generators["spectrum"] = lambda x: get_spectrum(resize_to_square(x))

    for size in _get_int(config.resize_sizes):
        spectrum_generators[f"resized_{size}"] = lambda x, s=size: get_spectrum(
            resize_to_square(x, s)
        )

    for size in _get_int(config.crop_sizes):
        sidelength = None
        if config.resize_before_crop:
            sidelength = 2 * size + 1
        spectrum_generators[f"cropped_{size}"] = (
            lambda x, s=size, side=sidelength: get_spectrum(
                resize_to_square(x, side=side)
            )[:s]
        )

    for size in _get_float(config.resize_sizes):
        spectrum_generators[f"resized_{size}"] = lambda x, s=size: get_spectrum(
            resize_to_square(x, _calculate_perc_sidelength(x, s))
        )

    for size in _get_float(config.crop_sizes):
        sidelength = None
        if config.resize_before_crop:
            sidelength = 2 * size + 1
        spectrum_generators[f"cropped_{size}"] = (
            lambda x, s=size, side=sidelength: get_spectrum(
                resize_to_square(x, side=side)
            )[: _calculate_perc_sidelength(x, s)]
        )

    return spectrum_generators


def psnr(x, xhat, rescale=True):
    """Compute Peak Signal to Noise Ratio in dB

    Inputs:
        x: Ground truth signal (range: [-1, 1])
        xhat: Reconstructed signal

    Outputs:
        snrval: PSNR in dB
    """

    # change data range from [-1, 1] to [0, 1]
    if rescale:
        x = (x + 1) / 2
        xhat = (xhat + 1) / 2

    err = x - xhat
    denom = np.mean(pow(err, 2))

    snrval = -10 * np.log10(denom)
    return snrval


def get_wasserstein(u, v):
    return wasserstein_distance(
        u_values=np.arange(len(u)), v_values=np.arange(len(u)), u_weights=u, v_weights=v
    )
