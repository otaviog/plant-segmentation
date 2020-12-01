"""
Training utilities
"""

from functools import partial
import typing

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from plantseg.data import SegmentationItem


class SegmentationBatch:
    """
    Contains the batch input and targets for the training forwarding.
    """

    def __init__(self, rgb_images: torch.Tensor, mask_images: torch.Tensor):
        self.rgb_images = rgb_images
        self.mask_images = mask_images

    @classmethod
    def create_from_preprocessing(cls, preproc_fun: typing.Callable,
                                  data_items: typing.Sequence[SegmentationItem]):
        """
        Create the batch using preprocessing function and segmentation
        dataset items.

        Args:

            preproc_fun: The network architecture preprocessing function.
            data_items: Items from the dataset.
        """

        preprocs = [preproc_fun(item.rgb_image, item.mask_image)
                    for item in data_items]

        rgb_images = torch.stack([
            torch.from_numpy(rgb_image).permute(2, 0, 1).float()
            for rgb_image, _ in preprocs])
        mask_images = torch.stack([
            torch.from_numpy(mask_image).permute(2, 0, 1).float()
            for _, mask_image in preprocs
        ])

        return cls(rgb_images, mask_images)

    def pin_memory(self):
        """
        Pin the memory.
        """
        self.rgb_images = self.rgb_images.pin_memory()
        self.mask_images = self.mask_images.pin_memory()
        return self

    def to(self, dst):
        """Sends to a device.
        """
        return SegmentationBatch(
            self.rgb_images.to(dst),
            self.mask_images.to(dst))

    def __getitem__(self, idx: int):
        return (self.rgb_images, self.mask_images)[idx]


def train_segmentation(
        model, preproc_fun, train_dataset, val_dataset, writer: SummaryWriter,
        batch_size=4, learning_rate=0.0001, max_epochs=40,
        device="cuda:0", verbose=True, num_workers=4):
    """
    Full training routine.
    """
    # pylint: disable=too-many-locals

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
    ])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    train_epoch = smp.utils.train.TrainEpoch(
        model, loss=loss, metrics=metrics,
        optimizer=optimizer,
        device=device, verbose=verbose)

    val_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics,
        device=device, verbose=verbose)

    collate_fn = partial(
        SegmentationBatch.create_from_preprocessing, preproc_fun)
    for iter_count in range(0, max_epochs):
        train_metrics = train_epoch.run(DataLoader(
            train_dataset, batch_size, collate_fn=collate_fn,
            num_workers=num_workers))
        val_metrics = val_epoch.run(DataLoader(
            val_dataset,
            batch_size, collate_fn=collate_fn,
            num_workers=num_workers,
            shuffle=False))
        writer.add_scalars("Dice Loss", {
            "dice_loss/train": train_metrics["dice_loss"],
            "dice_loss/val": val_metrics["dice_loss"]
        }, iter_count)

        writer.add_scalars("IoU Score", {
            "iou_score/train": train_metrics["iou_score"],
            "iou_score/val": val_metrics["iou_score"]
        }, iter_count)

        lr_scheduler.step(val_metrics["dice_loss"])
