from pathlib import Path
from dataclasses import dataclass

import torch
from natsort import natsorted
from PIL import Image
import numpy as np

import albumentations as A


@dataclass
class SegmentationItem:
    """SegmentationItem

    Attributes:
        anchor_rgb: Anchor rgb image.
        anchor_depth: Anchor depth image.
        kcam: intrinsic parameters.
        transform: [4x4] transformation matrix that
         transforms the anchor into the dock space.
        dock_rgb: Source rgb image.
        dock_depth: Source depth image.
    """
    rgb_image: np.ndarray
    mask_image: np.ndarray


def _load_plant_dir(plant_dir):
    rgb_images = plant_dir.glob("plant*_rgb.png")
    rgb_images = natsorted(rgb_images)
    seg_images = [plant_dir / (rgb_path.name.split('_')[0] + "_fg.png")
                  for rgb_path in rgb_images]

    return rgb_images, seg_images


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        super().__init__()
        base_path = Path(base_path)

        a1_rgb, a1_mask = _load_plant_dir(base_path / "A1")
        a2_rgb, a2_mask = _load_plant_dir(base_path / "A2")
        a3_rgb, a3_mask = _load_plant_dir(base_path / "A3")
        a4_rgb, a4_mask = _load_plant_dir(base_path / "A4")

        # self.rgb_images = a1_rgb + a2_rgb + a3_rgb + a4_rgb
        # self.seg_images = a1_mask + a2_mask + a3_mask + a4_mask

        self.rgb_images = a1_rgb + a2_rgb + a4_rgb
        self.seg_images = a1_mask + a2_mask + a4_mask

    def __getitem__(self, idx):
        rgb_image = np.array(Image.open(self.rgb_images[idx]))
        raw_mask_image = np.array(Image.open(
            self.seg_images[idx]).convert("L"))
        mask = raw_mask_image > 0

        mask_image = np.zeros_like(raw_mask_image, dtype=np.float32)
        mask_image[mask] = 1

        return SegmentationItem(rgb_image[:, :, :3], mask_image)

    def __len__(self):
        return len(self.rgb_images)


class AugmentationDataset(torch.utils.data.Dataset):
    """
    Augment the images.
    """

    def __init__(self, dataset,
                 crop_width, crop_height,
                 resize_width, resize_height):
        super().__init__()
        self.dataset = dataset
        self.augmentation = A.Compose([
            A.RandomCrop(width=crop_width, height=crop_height),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(resize_width, resize_height)
        ])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        transformed = self.augmentation(
            image=item.rgb_image, mask=item.mask_image)

        return SegmentationItem(
            transformed["image"], transformed["mask"])

    def __len__(self):
        return len(self.dataset)


class ResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, resize_width, resize_height):
        super().__init__()
        self.dataset = dataset
        self.augmentation = A.Compose([
            A.Resize(resize_width, resize_height)
        ])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        transformed = self.augmentation(
            image=item.rgb_image, mask=item.mask_image)

        return SegmentationItem(
            transformed["image"], transformed["mask"])

    def __len__(self):
        return len(self.dataset)
