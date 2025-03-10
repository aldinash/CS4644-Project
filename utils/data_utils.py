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

# normalize, add some noise for robustness and other shit like this
train_transforms = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomCropFromBorders(),
        A.OneOf(
            [
                A.ColorJitter(hue=0.1),
                A.Equalize(by_channels=False),
                A.FancyPCA(),
                A.GaussNoise(),
                A.ImageCompression(),
                A.RandomGamma(),
                A.RandomToneCurve(),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
                A.AdvancedBlur(),
                A.RandomBrightnessContrast(brightness_limit=[-0.3, 0.1]),
            ]
        ),
        A.Resize(520, 520),
        A.RandomRotate90(),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# similar shit
valid_transforms = A.Compose([A.Resize(520, 520), A.Normalize(), ToTensorV2()])

class SpacecraftDataset(Dataset):

    def __init__(
        self,
        transforms,
        path_to_data,
        split="train",
    ):
        if split == "train":
            self.image_path = path_to_data / "images/train"
            self.mask_path = path_to_data / "mask/train"
        else:
            self.image_path = path_to_data / "images/val"
            self.mask_path = path_to_data / "mask/val"

        self.image_filenames = sorted(self.image_path.glob("*.png"))
        self.transform = transforms

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Try/except is to skip to the next index for corrupted images
        or when a mask is absent
        """
        try:
            # Reading in an image
            image_filename = self.image_filenames[idx]
            image = cv2.imread(str(image_filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reading in a mask
            mask_filename = self.mask_path / str(
                image_filename.name.split(".")[0] + "_mask.png"
            )
            if mask_filename.is_file():
                mask = cv2.imread(str(mask_filename))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                raise (IOError)

            # Mask one-hot encoding: background + 3 classes
            mask[mask != 0] = 1
            foreground = np.ubyte(np.sum(mask, axis=2))
            background = foreground ^ 1
            background = background[:, :, np.newaxis]
            mask = np.concatenate((background, mask), axis=2).astype(np.float32)

            # Transformations/augmentations
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

            mask = mask.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            return image, mask.to(torch.float32)

        except:
            # skip to the next index
            if idx + 1 <= self.__len__():
                return self.__getitem__(idx + 1)
            else:
                raise StopIteration
