from pathlib import Path

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset

from config_utils import ExperimentConfig
from data_utils import load_from_file, get_mgrid, load_from_name


class Signal(Dataset):
    def __init__(
        self,
        signal: torch.Tensor,
        batch_size=None,
        test_percent=0.1,
    ):
        self.test_percent = test_percent
        self.channels = signal.shape[-1]
        self.resolution = signal.shape[:-1]
        self.dims = len(self.resolution)
        self.shape = signal.shape
        self.signal = signal
        self.signal_channel_first = torch.movedim(self.signal, -1, 0)
        self.signal_flat = self.signal.reshape(-1, self.channels)
        self.coords = get_mgrid(self.resolution, dim=self.dims)
        # self.coords_lr = get_mgrid(1000, dim=self.dims)

        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = np.prod(self.resolution)

        self.prep_train_test_split()
        self.train_pixels = self.signal_flat[self.train_idx]
        self.test_pixels = self.signal_flat[self.test_idx]
        self.train_coords = self.coords[self.train_idx]
        self.test_coords = self.coords[self.test_idx]

        self.prep_batches()
        self.batch_idx = -1

    def prep_train_test_split(self):
        idx = torch.randperm(self.signal_flat.shape[0])
        self.train_idx = idx[int(self.test_percent * len(idx)) :]
        self.test_idx = idx[: int(self.test_percent * len(idx))]

    def prep_batches(self):
        self.batch_idx = 0
        if len(self.train_coords) > self.batch_size:
            perm = torch.randperm(self.train_pixels.shape[0])
            self.batches_coords = self.train_coords[perm].split(self.batch_size)
            self.batches_pixels = self.train_pixels[perm].split(self.batch_size)
        else:
            self.batches_coords = self.train_coords.unsqueeze(0)
            self.batches_pixels = self.train_pixels.unsqueeze(0)

    def __getitem__(self, index):
        self.batch_idx += 1
        if self.batch_idx >= len(self.batches_coords):
            self.prep_batches()
        return self.batches_coords[self.batch_idx], self.batches_pixels[self.batch_idx]

    def __len__(self):
        return 1


class ImageDataset(Signal):
    def __init__(
        self,
        path,
        dataset_name,
        batch_size=None,
        test_percent=0.1,
    ):
        if path is not None:
            img = load_from_file(path).permute(1, 2, 0)
            self.name = Path(path).name.split(".")[0]
        elif dataset_name is not None:
            img = load_from_name(dataset_name).permute(1, 2, 0)
            self.name = dataset_name
        else:
            raise ValueError("Either path or dataset_name must be provided.")

        if len(img.shape) != 3:
            raise ValueError("Image must have 3 dimensions.")

        super().__init__(signal=img, batch_size=batch_size, test_percent=test_percent)
        self.img = self.signal_flat


class VideoDataset(Signal):
    def __init__(self, path: Path, batch_size: int, test_percent=0.1):
        self.name = Path(path).name.split(".")[0]
        if "npy" in path:
            self.vid = torch.tensor(np.load(path).astype(np.single))
        if "mp4" in path:
            # this is a fix required by skvideo (see https://github.com/scikit-video/scikit-video/issues/154)
            numpy.float = numpy.float64
            numpy.int = numpy.int_
            import skvideo.io

            self.vid = torch.tensor(skvideo.io.vread(str(path)).astype(np.single))
        self.vid = (self.vid / 255.0 * 2.0) - 1.0

        super().__init__(
            signal=self.vid, batch_size=batch_size, test_percent=test_percent
        )

    def prep_train_test_split(self):
        idx_img = torch.randperm(self.signal.shape[1] * self.signal.shape[2])
        idx_img_train = idx_img[int(self.test_percent * len(idx_img)) :]
        idx_img_test = idx_img[: int(self.test_percent * len(idx_img))]

        idx_vid = torch.arange(0, self.signal_flat.shape[0]).reshape(
            self.signal.shape[0], -1
        )
        self.idx_train = idx_vid[:, idx_img_train].flatten()
        self.idx_test = idx_vid[:, idx_img_test].flatten()


class DummyValidationDataset(Dataset):
    """Dummy dataset to trick PyTorchLightning."""

    def __getitem__(self, index):
        return 0, 0

    def __len__(self):
        return 1


def load_dataset(experiment_config: ExperimentConfig):
    if "mp4" in str(experiment_config.dataset_path) or "npy" in str(
        experiment_config.dataset_path
    ):
        dataset = VideoDataset(
            path=experiment_config.dataset_path,
            batch_size=experiment_config.batch_size,
        )
    else:
        dataset = ImageDataset(
            path=experiment_config.dataset_path,
            dataset_name=experiment_config.dataset_name,
            batch_size=experiment_config.batch_size,
        )
    return dataset
