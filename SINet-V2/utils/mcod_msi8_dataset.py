from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image


IMAGE_EXTENSIONS = {".npy", ".tif", ".tiff"}
MASK_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _collect_by_stem(root, extensions):
    files = {}
    for path in Path(root).iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def _paired_files(image_root, mask_root):
    images = _collect_by_stem(image_root, IMAGE_EXTENSIONS)
    masks = _collect_by_stem(mask_root, MASK_EXTENSIONS)
    stems = sorted(set(images) & set(masks))
    if not stems:
        raise RuntimeError("No MSI8 image/mask pairs found by filename stem.")
    return [(images[stem], masks[stem]) for stem in stems]


def _load_msi8(path):
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path))

    if arr.ndim != 3:
        raise ValueError(f"Expected 8-channel array, got shape {arr.shape} for {path}")
    if arr.shape[-1] == 8:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] != 8:
        raise ValueError(f"Expected HxWx8 or 8xHxW array, got shape {arr.shape} for {path}")

    arr = arr.astype(np.float32)
    max_value = float(arr.max()) if arr.size else 0.0
    if max_value > 1.0:
        arr /= 255.0 if max_value <= 255.0 else max_value
    return torch.from_numpy(np.clip(arr, 0.0, 1.0))


def _load_mask(path):
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    mask = (mask > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def _resize(image, mask, trainsize):
    image = F.interpolate(image.unsqueeze(0), size=(trainsize, trainsize), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask.unsqueeze(0), size=(trainsize, trainsize), mode="nearest")
    return image.squeeze(0), mask.squeeze(0)


class MCODMSI8Dataset(data.Dataset):
    def __init__(self, image_root, mask_root, trainsize, train=True):
        self.trainsize = trainsize
        self.train = train
        self.pairs = _paired_files(image_root, mask_root)
        self.size = len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = _load_msi8(image_path)
        mask = _load_mask(mask_path)
        image, mask = _resize(image, mask, self.trainsize)

        if self.train and torch.randint(0, 2, (1,)).item() == 1:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        return image, mask

    def __len__(self):
        return self.size


def get_loader(image_root, mask_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = MCODMSI8Dataset(image_root, mask_root, trainsize, train=True)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class test_dataset:
    def __init__(self, image_root, mask_root, testsize):
        self.testsize = testsize
        self.pairs = _paired_files(image_root, mask_root)
        self.size = len(self.pairs)
        self.index = 0

    def load_data(self):
        image_path, mask_path = self.pairs[self.index]
        image = _load_msi8(image_path)
        mask = Image.open(mask_path).convert("L").point(lambda p: 255 if p > 0 else 0)
        original_size = mask.size
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.testsize, self.testsize),
            mode="bilinear",
            align_corners=False,
        )
        name = image_path.stem + ".png"

        self.index += 1
        self.index = self.index % self.size

        return image, mask, name, original_size
