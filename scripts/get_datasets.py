import argparse
import shutil
from pathlib import Path

import deeplake
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# print avaiable datasets:
# print(sorted(deeplake.list('activeloop')))


def get_mgrid(sidelen=512):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    sidelen = 2 * (sidelen,)

    pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
        None, ...
    ].astype(np.float32)
    pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
    pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)

    pixel_coords -= 0.5
    pixel_coords *= 2 * np.pi

    return pixel_coords


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root", default="../data", type=str, help="Path to the dataset"
)
parser.add_argument(
    "--dataset",
    default="ffhq",
    type=str,
    choices=[
        "ffhq",
        "coco-train",
        "wiki-art",
        "chest-xray-train",
        "bikes",
        "sine",
        "kodak",
    ],
)
parser.add_argument("--num_images", default=10, type=int)
opt = parser.parse_args()
dataset_root = Path(opt.data_root)

is_deeplake = opt.dataset not in {"bikes", "sine", "kodak"}
if is_deeplake:
    ds = deeplake.load(f"hub://activeloop/{opt.dataset}", read_only=True)
    print(f"Dataset:\n{ds}\n", "-" * 20)

if opt.dataset == "ffhq":
    dataset_path = dataset_root / f"ffhq_1024_{opt.num_images}"
    dataset_path.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(opt.num_images)):
        img = ds.images_1024.image[i]
        img = Image.fromarray(img.numpy())
        img.save(dataset_path / f"{i}.png")

        for size in [128, 256, 512]:
            img_resized = img.resize((size, size), Image.LANCZOS)
            p = dataset_root / f"ffhq_{size}"
            p.mkdir(exist_ok=True, parents=True)
            img_resized.save(p / f"{i}.png")

    dataset_path = dataset_root / f"ffhq_wild_{opt.num_images}"
    dataset_path.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(opt.num_images)):
        img = ds.images_wild.image[i]
        img = Image.fromarray(img.numpy())
        img.save(dataset_path / f"{i}.png")
elif opt.dataset in {"coco-train", "wiki-art", "chest-xray-train"}:
    dataset_path = {
        "coco-train": dataset_root / f"coco_{opt.num_images}",
        "wiki-art": dataset_root / f"wiki_art_{opt.num_images}",
        "chest-xray-train": dataset_root / f"chest_xray_{opt.num_images}",
    }[opt.dataset]
    dataset_path.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(opt.num_images)):
        print(ds)
        img = ds.images[i].numpy()
        print(img.shape)
        if img.shape[-1] == 1:
            img = Image.fromarray(img.squeeze(-1), mode="L")
        else:
            img = Image.fromarray(img, mode="RGB")
        img.save(dataset_path / f"{i}.png")
elif opt.dataset == "kodak":
    dataset_path = dataset_root / f"kodak_{opt.num_images}"
    dataset_path.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(min(24, opt.num_images))):
        link = f"https://r0k.us/graphics/kodak/kodak/kodim{str(i + 1).zfill(2)}.png"
        img = Image.open(requests.get(link, stream=True).raw)
        img.save(dataset_path / f"{i}.png")

elif opt.dataset == "bikes":
    # this is a fix required by skvideo (see https://github.com/scikit-video/scikit-video/issues/154)
    np.float = np.float64
    np.int = np.int_
    import skvideo.io
    import skvideo.datasets

    (dataset_root / "video").mkdir(exist_ok=True, parents=True)
    path = Path(skvideo.datasets.bikes())
    shutil.copy(path, dataset_root / "video" / path.name)

    vid = skvideo.io.vread(str(path))
    np.save(dataset_root / "video" / f"{path.name.split('.')[0]}.npy", vid)
elif opt.dataset == "sine":

    sidelength = 512
    grid = get_mgrid(sidelength).squeeze(0)

    img = np.sin(grid[:, :, 1]) * 0.5 + 0.5
    print(img.min(), img.max())
    img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    img.save(dataset_root / f"sine_{sidelength}.png")
