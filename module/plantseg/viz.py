"""
Visualization functions
"""

import typing

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


class SegmentationViewer:
    """
    Viewer of segmentation datasets or results. It shows a slider
    for navigating the images and another one for adjusting the
    blending between background and foreground.
    """

    def __init__(self, dataset: typing.Union[
            torch.utils.data.Dataset, typing.List],
            title="Segmentation viewer",
            prob_threshold=0.5):
        """
        Setups the viewer.

        Args:

            dataset: Wrapped dataset.
            title: Window title
            prob_threshold: Probability threshold to whatever show or not
             the prediction.
        """
        self.dataset = dataset
        self.title = title
        self.canvas = None
        self.prob_threshold = prob_threshold

        self._last_item = None
        self._last_index = None

    def _get_dataset_item(self, idx):
        if self._last_index == idx:
            return self._last_item

        self._last_index = idx
        self._last_item = self.dataset[
            min(idx, len(self.dataset) - 1)]
        return self._last_item

    def _update_image(self, _):
        alpha = cv2.getTrackbarPos('Blending', self.title)/100

        item_idx = cv2.getTrackbarPos('Seek', self.title)
        item = self._get_dataset_item(item_idx)
        rgb_image = item.rgb_image

        self.canvas = rgb_image.copy()

        mask = item.mask_image.squeeze()

        heatmap = plt.get_cmap("plasma", 100)(mask)
        heatmap = (heatmap[:, :, :3]*255).astype(np.uint8)

        mask = mask > self.prob_threshold

        self.canvas[mask] = (rgb_image[mask]*(1.0 - alpha)
                             + heatmap[mask]*alpha).astype(np.uint8)
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)

    def run(self):
        """
        Shows and waits the viewer to quit.
        """

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('Seek', self.title, 0,
                           len(self.dataset),
                           self._update_image)
        cv2.createTrackbar('Blending', self.title, 50,
                           100, self._update_image)

        self._update_image(0)

        print("Press 'q' to close the window")
        while True:
            key = cv2.waitKey(10)
            key = chr(key & 0xff)

            if key == 'q':
                break

            cv2.imshow(self.title, self.canvas)
