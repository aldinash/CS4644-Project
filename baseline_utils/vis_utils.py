import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from tqdm import tqdm


def normim_to_numpy(image, mask, size=(720, 1280)):
    """Converts a normalized torch.Tensor image
    to a "H,W,C" array of integers [0-255]
    Restores the original image size (1280x720)
    """
    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                max_pixel_value=1.0,
            ),
            A.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0], max_pixel_value=1.0
            ),
            A.Resize(*size),
        ]
    )

    image = image.numpy().transpose((1, 2, 0))
    mask = mask[1:, :, :].numpy().transpose((1, 2, 0))
    transformed = transform(image=image, mask=mask)
    image = transformed["image"]
    mask = transformed["mask"]
    image = (image * 255).astype(np.uint8)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def predict_mask(model, image):
    """Predicting a mask"""
    model.eval()
    model.to("cpu")

    # predict logits [C, H, W]
    with torch.no_grad():
        pred_mask = model(image.unsqueeze(0))["out"].squeeze(0)

    # class for each pixel [H, W]
    pred_mask = pred_mask.argmax(dim=0)

    # one-hot [H, W, C]
    pred_mask = torch.eye(4)[pred_mask.to(torch.int64)]

    # background removed [H, W, 3 channels] == RGB image
    pred_mask = pred_mask[:, :, 1:].numpy()

    transform = A.Resize(720, 1280)
    image = image.numpy().transpose((1, 2, 0))
    pred_mask = transform(image=image, mask=pred_mask)["mask"]

    return pred_mask


def display_random_examples(dataset, n=2, model=None):
    if model is None:
        fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(10, 3 * n))
    else:
        fig, ax = plt.subplots(nrows=n, ncols=3, figsize=(15, 3 * n))

    for i in range(n):
        im_tensor, mask_tensor = dataset.__getitem__(np.random.randint(len(dataset)))
        image, mask = normim_to_numpy(im_tensor, mask_tensor)

        ax[i, 0].imshow(image)
        ax[i, 1].imshow(image)
        ax[i, 1].imshow(mask, alpha=0.5)

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Mask")

        if model is not None:
            ax[i, 2].imshow(image)
            ax[i, 2].imshow(predict_mask(model, im_tensor), alpha=0.5)
            ax[i, 2].set_title("Predicted mask")

    [axi.set_axis_off() for axi in ax.ravel()]
    plt.tight_layout()
    plt.show()
