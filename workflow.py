"""
Workflow for training image segmentation models
to segment plants from the Leaf Segmentation Challenge (LSC).

ci"""

import typing

import rflow

# pylint: disable=import-outside-toplevel,no-self-use


class LoadDataset(rflow.Interface):
    """
    Load a plant segmentation dataset in the 
    LEAF COUNTING CHALLENGE format.
    """

    def evaluate(self, resource: rflow.FSResource):
        """
        Load the dataset.

        Args:
            resource: The dataset base path.

        Returns: (obj:`plantseg.data.Dataset`):
            Load dataset.
        """
        from plantseg.data import Dataset

        dataset = Dataset(resource.filepath)

        self.save_measurement({"Size": len(dataset)})

        return dataset


class ViewSegmentationDataset(rflow.Interface):
    """
    View a segmentation dataset.
    """

    def evaluate(self, dataset):
        """
        Run the viewer.

        Args:

            dataset (List[plantseg.data.SegmentationItem]): Segmentation dataset.
        """
        from plantseg.viz import SegmentationViewer

        SegmentationViewer(dataset).run()


class SplitDataset(rflow.Interface):
    """
    Splits a dataset into train, validation and test sets.
    """

    def evaluate(self, resource: rflow.FSResource,
                 dataset, train_size, val_size, test_size):
        """
        Args:

            dataset (): A pytorch dataset
        """
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

        self.save_measurement({
            "Train size": train_size,
            "Val size": val_size,
            "Test size": test_size})

        return (Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices))

    def load(self, resource, dataset):
        """
        Load the splitted dataset.
        """
        from torch.utils.data import Subset
        train_indices, val_indices, test_indices = resource.pickle_load()

        return (Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices))


@rflow.graph()
def data(g):
    """
    Graph width dataset loading and split.
    """
    g.dataset = LoadDataset(rflow.FSResource(
        "dataset/plant-phenotyping/CVPPP2017_LCC_training/training"))

    g.dataset_split = SplitDataset(rflow.FSResource("dataset/dataset-split.pkl"))
    with g.dataset_split as args:
        args.dataset = g.dataset
        args.train_size = 0.9
        args.val_size = 0.05
        args.test_size = 0.05

    g.train_dataset_view = ViewSegmentationDataset()
    g.train_dataset_view.args.dataset = g.dataset_split[0]

    g.test_dataset_view = ViewSegmentationDataset()
    g.test_dataset_view.args.dataset = g.dataset_split[2]


class AugmentDataset(rflow.Interface):
    """
    Create a dataset with random augmentations
    """

    def evaluate(self, dataset, crop_width: int, crop_height: int):
        """
        Factory the augmentated dataset.

        Args:
            dataset (:obj:`torch.data.utils.Dataset`): Any 
             segmentation item dataset.
            crop_width: Augmentation crop width.
            crop_height: Augmentation crop height.
        """
        from plantseg.data import AugmentationDataset

        return AugmentationDataset(dataset, crop_width, crop_height)


class Train(rflow.Interface):
    """Segmentation training task.
    """

    def non_collateral(self):
        return ["device", "verbose", "num_workers"]

    def evaluate(self, resource: rflow.FSResource, model, preprocessing: typing.Callable,
                 train_dataset, val_dataset,
                 batch_size=4,
                 learning_rate=0.0001,
                 max_epochs=40,
                 device="cuda:0", verbose=True,
                 num_workers=4):
        """
        Runs the training.

        Args:
            resource: Where to output the torch model.
            model: Model to be trained.
            preprocessing: The architecture's preprocessing.
            train_dataset (:obj:`torch.data.utils.Dataset`):
             Training dataset of segmentation items.
            val_dataset (:obj:`torch.data.utils.Dataset`):
             Validation dataset of segmentation items.
            ...
        """
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from plantseg.config import get_expid
        from plantseg.train import train_segmentation

        exp_id = get_expid("runs/")
        writer = SummaryWriter("runs/" + exp_id)
        print("Experiment ID: ", exp_id)
        train_segmentation(
            model, preprocessing, train_dataset, val_dataset,
            writer, batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            device=device, verbose=verbose,
            num_workers=num_workers)

        torch.save(model.state_dict(), resource.filepath)

        return model.eval()

    def load(self, resource, model):
        """
        Load the model with its previosly trained weights.
        """
        import torch

        model.load_state_dict(torch.load(resource.filepath))
        return model.eval()


class ViewDatasetPredictions(rflow.Interface):
    """Opens a window to visualize a model predictions
    on a dataset.
    """

    def non_collateral(self):
        """Non collateral arguments"""
        return ["device"]

    def evaluate(self, dataset, model, preprocessing,
                 device="cuda:0"):
        from plantseg.viz import SegmentationViewer
        from plantseg.inference import (
            SegmentationPredictor, PredictDataset)

        predictor = SegmentationPredictor(
            model.to(device).eval(),
            preprocessing)

        pred_dataset = PredictDataset(dataset, predictor)
        SegmentationViewer(pred_dataset).run()


class ViewImagePrediction(rflow.Interface):
    """Opens a window to visualize a single image
    prediction.
    """

    def non_collateral(self):
        """Non collateral arguments"""
        return ["device"]

    def evaluate(self, image_path, model, preprocessing,
                 device="cuda:0"):
        """
        Args:
            image_path: An image input file path.
        """
        from PIL import Image
        import numpy as np

        from plantseg.inference import SegmentationPredictor
        from plantseg.viz import SegmentationViewer
        from plantseg.data import SegmentationItem

        predictor = SegmentationPredictor(model.to(device).eval(),
                                          preprocessing)

        image = np.array(Image.open(image_path))[:, :, :3]
        mask = predictor.predict(image)

        SegmentationViewer([SegmentationItem(image, mask)]).run()


class EvaluateMetrics(rflow.Interface):
    def non_collateral(self):
        return ["batch_size", "device", "num_workers"]

    def evaluate(self, dataset, model, preprocessing,
                 prob_threshold=0.5,
                 batch_size=4,
                 device="cuda:0", num_workers=4):
        from functools import partial
        import segmentation_models_pytorch as smp
        from torch.utils.data import DataLoader

        from plantseg.train import SegmentationBatch

        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=prob_threshold),
            smp.utils.metrics.Accuracy(threshold=prob_threshold),
            smp.utils.metrics.Precision(threshold=prob_threshold),
            smp.utils.metrics.Recall(threshold=prob_threshold)]

        collate_fn = partial(
            SegmentationBatch.create_from_preprocessing, preprocessing)
        val_epoch = smp.utils.train.ValidEpoch(
            model, loss=loss, metrics=metrics,
            device=device, verbose=True)
        val_metrics = val_epoch.run(DataLoader(
            dataset,
            batch_size, collate_fn=collate_fn,
            num_workers=num_workers,
            shuffle=False))
        print(val_metrics)
        self.save_measurement(val_metrics)


def _make_experiment(g, model_node,
                     crop_width, crop_height):
    data_g = rflow.open_graph(".", "data")

    g.train_dataset = AugmentDataset(show=False)
    with g.train_dataset as args:
        args.dataset = data_g.dataset_split[0]
        args.crop_width = crop_width
        args.crop_height = crop_height

    g.train_dataset_view = ViewSegmentationDataset()
    g.train_dataset_view.args.dataset = g.train_dataset

    val_dataset = data_g.dataset_split[1]
    test_dataset = data_g.dataset_split[2]

    model = model_node[0]
    preprocessing = model_node[1]

    g.untrain_test = ViewDatasetPredictions()
    with g.untrain_test as args:
        args.dataset = test_dataset
        args.model = model
        args.preprocessing = preprocessing

    g.train = Train(rflow.FSResource(f"{g.name}.torch"))
    with g.train as args:
        args.model = model
        args.preprocessing = preprocessing
        args.train_dataset = g.train_dataset
        args.val_dataset = val_dataset
        args.batch_size = 8
        args.learning_rate = 0.001
        args.max_epochs = 50
        args.num_workers = rflow.UserArgument(
            "--num_workers", default=4, type=int)

    g.predict_test_set = ViewDatasetPredictions()
    with g.predict_test_set as args:
        args.dataset = test_dataset
        args.model = g.train
        args.preprocessing = preprocessing

    g.predict_image = ViewImagePrediction()
    with g.predict_image as args:
        args.image_path = rflow.UserArgument(
            "--sample-image",
            default="dataset/resources/49592690973_35154a72d4_b.jpg")
        args.model = g.train
        args.preprocessing = preprocessing

    g.metrics = EvaluateMetrics()
    with g.metrics as args:
        args.dataset = data_g.dataset_split[2]
        args.model = g.train
        args.preprocessing = preprocessing


class CreateModel(rflow.Interface):
    """
    Node to factory segmentation_models_pytorch models.
    """

    def evaluate(self, architecture="FPN", encoder="se_resnext50_32x4d",
                 encoder_weights="imagenet", activation="sigmoid",
                 resize_width=224, resize_height=224):
        import segmentation_models_pytorch as smp

        from plantseg.inference import Preprocessing

        if architecture == "FPN":
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=1,
                activation=activation)
        elif architecture == "UNet":
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=1,
                activation=activation)
        else:
            raise RuntimeError(f"Undefined architecture {architecture}")
        preproc_fun = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)

        return model, Preprocessing(preproc_fun, resize_width, resize_height)


@rflow.graph()
def fpn(g):
    g.model = CreateModel()
    _make_experiment(g, g.model, 256, 256)


@rflow.graph()
def fpn_resnet(g):
    g.model = CreateModel()
    with g.model as args:
        args.encoder = "resnet34"

    _make_experiment(g, g.model, 256, 256)


@rflow.graph()
def unet(g):
    g.model = CreateModel()
    with g.model as args:
        args.architecture = "UNet"
        args.encoder = "resnet34"

    _make_experiment(g, g.model, 224, 224)
