import random
from pathlib import Path

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

try:
    from torchvision.transforms import InterpolationMode
    MASK_INTERPOLATION = InterpolationMode.NEAREST
except ImportError:
    MASK_INTERPOLATION = Image.NEAREST


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _collect_by_stem(root):
    files = {}
    for path in Path(root).iterdir():
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
            files[path.stem] = path
    return files


def _paired_files(image_root, mask_root):
    images = _collect_by_stem(image_root)
    masks = _collect_by_stem(mask_root)
    stems = sorted(set(images) & set(masks))
    if not stems:
        raise RuntimeError("No image/mask pairs found by filename stem.")
    return [(images[stem], masks[stem]) for stem in stems]


def cv_random_flip(img, label):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width, image_height = image.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, Image.BICUBIC)
        label = label.rotate(random_angle, Image.NEAREST)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


class MCODDataset(data.Dataset):
    def __init__(self, image_root, mask_root, trainsize):
        self.trainsize = trainsize
        self.pairs = _paired_files(image_root, mask_root)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=MASK_INTERPOLATION),
            transforms.ToTensor(),
        ])
        self.size = len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = self.rgb_loader(image_path)
        mask = self.binary_loader(mask_path)

        image, mask = cv_random_flip(image, mask)
        image, mask = randomCrop(image, mask)
        image, mask = randomRotation(image, mask)
        image = colorEnhance(image)

        return self.img_transform(image), self.gt_transform(mask)

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).convert("L")
            return img.point(lambda p: 255 if p > 0 else 0)

    def __len__(self):
        return self.size


def get_loader(image_root, mask_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = MCODDataset(image_root, mask_root, trainsize)
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
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.size = len(self.pairs)
        self.index = 0

    def load_data(self):
        image_path, mask_path = self.pairs[self.index]
        image = self.rgb_loader(image_path)
        image = self.transform(image).unsqueeze(0)
        mask = self.binary_loader(mask_path)
        name = image_path.stem + ".png"
        image_for_post = self.rgb_loader(image_path).resize(mask.size)

        self.index += 1
        self.index = self.index % self.size

        return image, mask, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).convert("L")
            return img.point(lambda p: 255 if p > 0 else 0)
