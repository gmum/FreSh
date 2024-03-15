from pathlib import Path
from typing import Union, Tuple

import numpy as np
import skimage
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.transforms.functional import resize


def load_from_file(path: Union[str, Path], resolution=None) -> torch.Tensor:
    Image.MAX_IMAGE_PIXELS = 933120000  # needed to load huge images
    img = Image.open(path).convert("RGB")

    if resolution is None:
        resolution = (img.height, img.width)
    transform = Compose(
        [
            Resize(resolution),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
        ]
    )
    return transform(img)


def load_from_name(name: Union[str, Path], resolution=None) -> torch.Tensor:
    """Load scikit datasets."""
    if name == "camera":
        img = Image.fromarray(skimage.data.camera())
    elif name == "grass":
        img = Image.fromarray(skimage.data.grass())
    elif name == "mitosis":
        img = Image.fromarray(skimage.data.human_mitosis())
    else:
        raise ValueError(f"Unknown dataset name '{name}'")

    if resolution is None:
        resolution = (img.height, img.width)
    transform = Compose(
        [
            Resize(resolution),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
        ]
    )
    return transform(img)


def get_mgrid(sidelen: Union[int, Tuple], dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def resize_to_square(img: torch.tensor, side=None):
    sides = torch.tensor(img.shape[1:])
    if (
        torch.all(sides == sides[0])
        and side is None
        and sides[0] <= 1024
        and sides[1] <= 1024
    ):
        return img
    if side is None:
        side = sides.min()
    side = min(side, 1024)
    if len(sides) == 2:
        return resize(img, [side, side], antialias=True)
    elif len(sides) == 3:
        return interpolate(
            img.unsqueeze(0), [side] * len(sides), mode="trilinear"
        ).squeeze(0)


def tensor_to_image(tensor, limit_size=True, rescale=True):
    if len(tensor.shape) == 4:
        # extract first frame from video
        tensor = tensor[:, 0]

    # channel is last
    size_limit = 512
    if limit_size and (tensor.shape[-1] > size_limit or tensor.shape[-2] > size_limit):
        # decreasing memory needed by logs
        max_side = max(tensor.shape[-1], tensor.shape[-2])
        res = [
            int(tensor.shape[-2] / max_side * size_limit),
            int(tensor.shape[-1] / max_side * size_limit),
        ]
        tensor = resize(tensor, res, antialias=True)

    arr = torch.movedim(tensor, 0, -1).numpy().clip(-1, 1)
    if rescale:
        arr = arr * 0.5 + 0.5

    if arr.shape[-1] == 3:
        return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    else:
        return Image.fromarray((arr * 255).astype(np.uint8).squeeze(-1), mode="L")
