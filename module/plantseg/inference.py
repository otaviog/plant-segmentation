"""
Segmentation inference tools.
"""
import typing

import torch
import torchvision.transforms as T
import numpy as np
import albumentations

from plantseg.data import SegmentationItem


class Preprocessing:
    """Wrapper for resizing and executing the segmentation_models_pytorch
    preprocessing function.
    """

    def __init__(self, smp_preproc_fun: typing.Callable,
                 resize_width: int, resize_height: int):
        self.smp_preproc_fun = smp_preproc_fun
        self.resize = albumentations.Resize(resize_width, resize_height)

    def __call__(self, image, mask=None):
        transformed = self.resize(image=image, mask=mask)
        image = self.smp_preproc_fun(transformed["image"])
        mask = transformed["mask"]

        if mask is None:
            return image

        return image, mask


class SegmentationPredictor:
    """
    Facade for running the model prediction.
    """

    def __init__(self, model: torch.nn.Module,
                 preproc_fun: typing.Callable):
        """
        Args:

            model: Segmentation model.
            preproc_fun: Architecture's preprocessing function.
            resize_width: Resize width.
            resize_height: Resize height.
        """
        self.model = model
        self.preproc_fun = preproc_fun

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Run prediction for one image.
        """
        height, width = rgb_image.shape[:2]
        rgb_image = self.preproc_fun(rgb_image)
        rgb_tensor = torch.from_numpy(
            rgb_image).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mask_image = self.model(rgb_tensor)

        mask_image = torch.nn.functional.interpolate(
            mask_image.cpu(), (height, width))

        return mask_image.squeeze().numpy()

    @property
    def device(self):
        """Model's current device"""
        return next(iter(self.model.parameters())).device


class PredictDataset(torch.utils.data.Dataset):
    """
    Dataset for image segmentation items that runs
    a predictor to set the masks.
    """

    def __init__(self, dataset: torch.utils.data.Dataset,
                 predictor: SegmentationPredictor):
        """
        Args:

            dataset: The input dataset.
            predictor: The predictor instance.
        """
        self.dataset = dataset
        self.predictor = predictor

    def __getitem__(self, idx: int) -> SegmentationItem:
        item = self.dataset[idx]
        pred = self.predictor.predict(item.rgb_image)
        return SegmentationItem(item.rgb_image, pred)

    def __len__(self):
        return len(self.dataset)
