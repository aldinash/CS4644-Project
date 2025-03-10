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


def IoU(preds, targs, device="cuda", eps: float = 1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Rewritten from https://github.com/kevinzakka/pytorch-goodies
        for one-hot-encoded masks (targs)

    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, C, H, W]. C==0 is the background class
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value
             for multi-class image segmentation
    """

    # Take softmax along class dimension;
    # all class probs add to 1 (per pixel)
    probas = F.softmax(preds, dim=1)

    targs = targs.type(preds.type()).to(device)

    # Sum probabilities by class and across batch images
    # Background class (dims==1) is excluded (dims==0 is for batch)
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * targs, dims)  # [class0,class1,...]
    cardinality = torch.sum(probas + targs, dims)  # [class0,class1,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()  # find mean class IoU values
    return iou


class SegModel(pl.LightningModule):
    def __init__(self, model, batch_size, lr, train_dataset, valid_dataset):
        super(SegModel, self).__init__()
        self.net = model
        self.batch_size = batch_size
        self.learning_rate = lr
        self.trainset = train_dataset
        self.valset = valid_dataset
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        out = self.forward(img)["out"]  # [B, C, H, W]
        loss_val = self.criterion(out, mask.float())
        iou_score = IoU(out, mask)
        self.log("train_loss", loss_val, prog_bar=True)
        self.log("train_iou", iou_score, prog_bar=True)
        return loss_val

    def configure_optimizers(self):
        opt = torch.optim.Adamax(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            drop_last=True,  # drop last incomplete batch
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            drop_last=True,  # drop last incomplete batch
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        out = self.forward(img)["out"]
        loss_val = self.criterion(out, mask.float())
        iou_score = IoU(out, mask)
        self.log("val_loss", loss_val, prog_bar=True)
        self.log("val_iou", iou_score, prog_bar=True)
