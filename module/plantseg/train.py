"""
Training utilities
"""

from functools import partial

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


from plantseg.data import ResizeDataset


class SegmentationBatch:
    def __init__(self, rgb_images, mask_images):
        self.rgb_images = rgb_images
        self.mask_images = mask_images

    @classmethod
    def create_from_preprocessing(cls, preproc_fun, data_items):
        rgb_images = torch.stack([
            torch.from_numpy(
                preproc_fun(item.rgb_image)).permute(2, 0, 1).float()
            for item in data_items])
        mask_images = torch.stack([
            torch.from_numpy(item.mask_image) for item in data_items])

        return cls(rgb_images, mask_images)

    def pin_memory(self):
        self.rgb_images = self.rgb_images.pin_memory()
        self.mask_images = self.mask_images.pin_memory()
        return self

    def to(self, dst):
        return SegmentationBatch(
            self.rgb_images.to(dst),
            self.mask_images.to(dst))

    def __getitem__(self, idx):
        return (self.rgb_images, self.mask_images)[idx]


def train_segmentation(
        model, preproc_fun, train_dataset, val_dataset, writer,
        batch_size=4, learning_rate=0.0001, max_epochs=40,
        device="cuda:0", verbose=True, num_workers=4):
    """
    Full training routine.
    """
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
    ])
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
