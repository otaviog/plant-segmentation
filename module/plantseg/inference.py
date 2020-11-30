"""
Segmentation inference tools.
"""
import torch
import torchvision.transforms as T

from plantseg.data import SegmentationItem


class PredictDataset:
    def __init__(self, dataset, model, preproc_fun, resize_width, resize_height):
        self.dataset = dataset
        self.model = model
        self.preproc_fun = preproc_fun
        self.resize_width = resize_width
        self.resize_height = resize_height

    def predict(self, rgb_image):
        rgb_image = self.preproc_fun(rgb_image)
        rgb_tensor = torch.from_numpy(
            rgb_image).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        rgb_tensor = T.functional.resize(
            rgb_tensor, (self.resize_height, self.resize_width))

        with torch.no_grad():
            mask_image = self.model(rgb_tensor)

        mask_image = T.functional.resize(
            mask_image.squeeze(0), (rgb_image.shape[0], rgb_image.shape[1]))
        return mask_image.squeeze()

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def __getitem__(self, idx):

        item = self.dataset[idx]
        pred = self.predict(item.rgb_image)
        return SegmentationItem(item.rgb_image, pred.cpu().numpy())

    def __len__(self):
        return len(self.dataset)
