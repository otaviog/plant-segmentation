import torch
import segmentation_models_pytorch as smp
import cv2

import rflow


class LoadDataset(rflow.Interface):
    def evaluate(self, resource):
        from plantseg.data import Dataset

        return Dataset(resource.filepath)


class ConcatDatasets(rflow.Interface):
    def evaluate(self, dataset1, dataset2, dataset3=None):
        from torch.utils.data import ConcatDataset

        return ConcatDataset(
            [dataset for dataset in (dataset1, dataset2, dataset3)
             if dataset is not None])


class ViewSegmentationDataset(rflow.Interface):
    def evaluate(self, dataset):
        from plantseg.viz import SegmentationViewer

        SegmentationViewer(dataset).run()


class SplitDataset(rflow.Interface):
    def evaluate(self, resource, dataset, train_size, val_size, test_size):
        import random
        from torch.utils.data import Subset

        size = len(dataset)
        indices = list(range(size))
        random.shuffle(indices)

        train_size = int(train_size*size)
        val_size = int(val_size*size)
        test_size = int(test_size*size)

        train_indices = indices[:train_size]
        train_val_size = train_size + val_size

        val_indices = indices[train_size:train_val_size]
        test_indices = indices[train_val_size:min(
            train_val_size + test_size, size)]

        resource.pickle_dump([train_indices, val_indices, test_indices])

        return (Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices))

    def load(self, resource, dataset):
        from torch.utils.data import Subset
        train_indices, val_indices, test_indices = resource.pickle_load()

        return (Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices))


@rflow.graph()
def data(g):
    g.dataset = LoadDataset(rflow.FSResource(
        "dataset/plant-phenotyping/CVPPP2017_LCC_training/training"))

    g.dataset_split = SplitDataset(rflow.FSResource("dataset-split.pkl"))
    with g.dataset_split as args:
        args.dataset = g.dataset
        args.train_size = 0.8
        args.val_size = 0.1
        args.test_size = 0.1

    g.train_dataset_view = ViewSegmentationDataset()
    g.train_dataset_view.args.dataset = g.dataset_split[0]

    g.test_dataset_view = ViewSegmentationDataset()
    g.test_dataset_view.args.dataset = g.dataset_split[2]


class AugmentDataset(rflow.Interface):
    def evaluate(self, dataset):
        from plantseg.data import AugmentationDataset

        return AugmentationDataset(dataset)


class SegmentationBatch:
    def __init__(self, rgb_images, mask_images):
        self.rgb_images = rgb_images
        self.mask_images = mask_images

    @classmethod
    def create_from_preprocessing(cls, preproc_fun, data_items):
        rgb_images = torch.stack([
            torch.from_numpy(preproc_fun(
                cv2.resize(item.rgb_image, (224, 224)))).permute(2, 0, 1).float()
            for item in data_items])
        mask_images = torch.stack([
            torch.from_numpy(cv2.resize(item.mask_image, (224, 224)))
            for item in data_items])

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


class Train(rflow.Interface):
    def non_collateral(self):
        return ["device", "verbose", "num_workers"]

    def evaluate(self, resource, model, preproc_fun,
                 train_dataset, val_dataset,
                 batch_size=4,
                 learning_rate=0.0001,
                 max_epochs=40,
                 device="cuda:0", verbose=True,
                 num_workers=4):
        from functools import partial

        import torch
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard import SummaryWriter
        import segmentation_models_pytorch as smp

        from plantseg.config import get_expid

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

        writer = SummaryWriter("runs/" + get_expid("runs/"))
        collate_fn = partial(
            SegmentationBatch.create_from_preprocessing, preproc_fun)
        for iter_count in range(0, max_epochs):
            train_metrics = train_epoch.run(DataLoader(
                train_dataset, batch_size, collate_fn=collate_fn,
                num_workers=num_workers))
            val_metrics = val_epoch.run(DataLoader(
                val_dataset, batch_size, collate_fn=collate_fn,
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

        torch.save(model.state_dict(), resource.filepath)

        return model.eval()

    def load(self, resource, model):
        import torch

        model.load_state_dict(torch.load(resource.filepath))
        return model.eval()


def _make_experiment(g, model_node):
    data_g = rflow.open_graph(".", "data")

    g.train_dataset = AugmentDataset()
    g.train_dataset.args.dataset = data_g.dataset_split[0]

    g.train_dataset_view = ViewSegmentationDataset()
    g.train_dataset_view.args.dataset = g.train_dataset

    g.train = Train(rflow.FSResource("fpn1.torch"))
    with g.train as args:
        args.model = model_node[0]
        args.preproc_fun = model_node[1]
        args.train_dataset = g.train_dataset
        args.val_dataset = data_g.dataset_split[1]
        args.max_epochs = 10


class CreateModel(rflow.Interface):
    def evaluate(self, encoder="se_resnext50_32x4d",
                 encoder_weights="imagenet",
                 activation="sigmoid"):
        import segmentation_models_pytorch as smp

        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=1,
            activation=activation)

        preproc_fun = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)

        return model, preproc_fun


@rflow.graph()
def fpn(g):
    g.model = CreateModel()
    _make_experiment(g, g.model)


@rflow.graph()
def fpn_resnet(g):
    g.model = CreateModel()
    with g.model as args:
        args.encoder = "resnet34"

    _make_experiment(g, g.model)
