import argparse
import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from torchvision.transforms.functional import resize


def _get_int(sizes) -> List[int]:
    return list(filter(lambda x: isinstance(x, int), sizes))


def _get_float(sizes) -> List[float]:
    return list(filter(lambda x: isinstance(x, float), sizes))


def _calculate_perc_sidelength(img, percentage):
    sidelength = torch.tensor(img.shape[1:]).min()
    return int(sidelength * percentage)


def prepare_argparser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--model_output", default=None, type=str)
    parser.add_argument("--results_root", default="./wasserstein_distances", type=str)
    parser.add_argument("--verbose", action="store_true")
    # parser.add_argument("--channel_first", action="store_true")
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--downscale", action="store_true")
    return parser


@dataclasses.dataclass(frozen=True, order=False)
class SpectrumConfig:
    use_baseline: bool
    eval_before_training: bool
    resize_sizes: List[Union[int, float]]
    crop_sizes: List[Union[int, float]]
    selection_methods: List[str]
    initial_evals: int


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
    return resize(img, [side, side], antialias=True)


def get_fft(y: torch.Tensor, dim=None):
    """Calculate DFT of y.

    Inputs:
    - y: Tensor of shape (C, ...)
    - dim: dimensions to calculate DFT over.

    Outputs:
    - y_fft: Absolute values of DFT of y, summed across C.
    """
    if y.ndim == 2:
        y = y[None, :, :]
    if dim is None:
        dim = tuple(range(1, y.ndim))

    if y.ndim not in {3, 4}:
        raise ValueError("y should have 3 or 4 dimensions.")

    y_fft = torch.fft.fftn(y, dim=dim)
    return torch.abs(y_fft).sum(dim=0)


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
    if drop_bias:
        fft_y = get_fft(y)
        return _spectrum_from_fft(fft_y)[1:]
    else:
        return _spectrum_from_fft(get_fft(y))


def prepare_spectrum_generators(config: SpectrumConfig):
    if np.any([p <= 0 or p > 1 for p in _get_float(config.resize_sizes)]):
        raise ValueError("Percentage sizes should be in (0, 1]")

    spectrum_generators = dict()
    if config.use_baseline:
        spectrum_generators["spectrum"] = lambda x: get_spectrum(resize_to_square(x))

    for size in _get_int(config.crop_sizes):
        spectrum_generators[f"{size}"] = lambda x, s=size: get_spectrum(
            resize_to_square(x)
        )[:s]

    for size in _get_float(config.crop_sizes):
        spectrum_generators[f"{size}"] = lambda x, s=size: get_spectrum(
            resize_to_square(x)
        )[: _calculate_perc_sidelength(x, s)]

    return spectrum_generators


def get_wasserstein(u, v):
    return wasserstein_distance(
        u_values=np.arange(len(u)), v_values=np.arange(len(u)), u_weights=u, v_weights=v
    )


if __name__ == "__main__":
    parser = prepare_argparser()
    opt, extras = parser.parse_known_args()

    spectrum_generators = prepare_spectrum_generators(
        SpectrumConfig(
            use_baseline=False,
            eval_before_training=False,
            resize_sizes=[],
            crop_sizes=[32, 64, 128],
            selection_methods=[],
            initial_evals=0,
        )
    )

    if opt.verbose:
        print(f"Loading dataset... ({opt.dataset})")
    dataset = np.load(opt.dataset)
    if len(dataset.shape) == 3:
        dataset = dataset[None, ...]

    # dataset format: (N, H, W, C); for images N=1, for videos/NeRF N>1
    if dataset.shape[-1] <= 3:
        dataset = np.moveaxis(dataset, -1, 1)

    if opt.verbose:
        print("Dataset loaded...")

    dataset_name = Path(opt.dataset).stem
    np.random.seed(0)
    results_file = Path(opt.results_root) / f"{dataset_name}/log.txt"
    results_file.parent.mkdir(exist_ok=True, parents=True)
    with open(results_file, "w") as f:
        results = dict()
        results_all = defaultdict(list)
        for name, spectrum_fun in spectrum_generators.items():
            best_distance = np.inf
            best_se = None
            best_config = None
            dataset_idx = np.random.choice(
                dataset.shape[0], size=10, replace=dataset.shape[0] < 10
            )
            bboxes = None
            if opt.bbox:
                gt_imgs = dataset[dataset_idx]
                bkg_masks = (gt_imgs == 0).sum(axis=1) == 3
                bboxes = np.empty((gt_imgs.shape[0], 4), dtype=np.int32)
                for i in range(gt_imgs.shape[0]):
                    bboxes[i, 0] = np.min(np.where(~bkg_masks[i])[0])
                    bboxes[i, 1] = np.max(np.where(~bkg_masks[i])[0]) + 1
                    bboxes[i, 2] = np.min(np.where(~bkg_masks[i])[1])
                    bboxes[i, 3] = np.max(np.where(~bkg_masks[i])[1]) + 1

            for folder in sorted(Path(opt.model_output).iterdir()):
                if not folder.is_dir():
                    continue
                if opt.verbose:
                    print("Dataset shape", dataset.shape, file=f)

                distances = []
                if opt.verbose:
                    print("Computing distances...", file=f)

                model_out_files = list(
                    filter(lambda x: "npy" in str(x), folder.iterdir())
                )
                if len(model_out_files) == 0:
                    continue

                n_plots = min(10, (len(model_out_files) + 1) // 2)
                fig, axs = plt.subplots(2, n_plots, figsize=(15, 5))
                axs = axs.flatten()

                for i, file in enumerate(model_out_files):
                    out = np.load(file)
                    idx = dataset_idx[i]

                    if len(out.shape) > 3:
                        idx = np.random.choice(out.shape[0])
                        out = out[idx].copy()
                    if out.shape[-1] <= 3:
                        out = np.moveaxis(out, -1, 0)
                    gt_img = dataset[idx].copy()
                    if opt.downscale:
                        size = int(name.split("_")[-1])
                        out = resize_to_square(torch.tensor(out), 2 * size + 1).numpy()
                        gt_img = resize_to_square(
                            torch.tensor(gt_img), 2 * size + 1
                        ).numpy()

                    if bboxes is not None:
                        gt_img = gt_img[
                            :, bboxes[i, 0] : bboxes[i, 1], bboxes[i, 2] : bboxes[i, 3]
                        ]
                        out = out[
                            :, bboxes[i, 0] : bboxes[i, 1], bboxes[i, 2] : bboxes[i, 3]
                        ]

                    gt_spec = spectrum_fun(torch.Tensor(gt_img))
                    out_spec = spectrum_fun(torch.Tensor(out))

                    w = get_wasserstein(gt_spec, out_spec)
                    distances.append(w)

                    if i < len(axs):
                        axs[i].plot(gt_spec / sum(gt_spec), label="gt")
                        axs[i].plot(out_spec / sum(out_spec), label="out")

                axs[0].legend()
                p = Path(opt.results_root) / f"{dataset_name}/plots/{name}"
                p.mkdir(exist_ok=True, parents=True)
                plt.savefig(p / f"{folder.name}.png")
                plt.close(fig)

                mean_distance = np.mean(distances)
                se = np.std(distances) / np.sqrt(len(distances))
                if mean_distance < best_distance:
                    best_distance = mean_distance
                    best_config = folder.name
                    best_se = se

                results_all[name].append((folder.name, mean_distance, best_se))

                print(f"*** {folder.name} ***", file=f)
                if opt.verbose:
                    print(
                        f"[{name}] distances are {[round(x, 2) for x in distances]}",
                        file=f,
                    )
                print(
                    f"[{name}] mean Wasserstein is {mean_distance:.4f} (SE={se:.4f})\n\n",
                    file=f,
                )

            print(f"Best config is '{best_config}' (W={best_distance:.4f})\n\n", file=f)
            results[name] = (best_config, round(best_distance, 2), best_se)

        csv_file_best = Path(opt.results_root) / f"wasserstein_best.csv"
        csv_file_all = Path(opt.results_root) / f"wasserstein_all.csv"
        try:
            df = pd.read_csv(csv_file_best)
            df_all = pd.read_csv(csv_file_all)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "dataset",
                    "spectrum_size",
                    "wasserstein",
                    "wasserstein_standard_error",
                ]
            )
            df_all = pd.DataFrame(
                columns=[
                    "dataset",
                    "spectrum_size",
                    "config",
                    "wasserstein",
                    "wasserstein_standard_error",
                ]
            )
        for name, best in results.items():
            print(name, best, file=f)
            row = {
                "dataset": dataset_name,
                "spectrum_size": name,
                "best_configuration": best[0],
                "wasserstein": best[1],
                "wasserstein_standard_error": best[2],
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv_file_best, index=False)

        for name, all in results_all.items():
            data = {
                "dataset": [dataset_name] * len(all),
                "method": [name] * len(all),
                "config": [x[0] for x in all],
                "wasserstein": [x[1] for x in all],
                "standard_error": [x[2] for x in all],
            }
            df_all = pd.concat([df_all, pd.DataFrame(data)], ignore_index=True)
            df_all.to_csv(csv_file_all, index=False)
