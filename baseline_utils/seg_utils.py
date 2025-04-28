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

def segmentation_total_loss(pred_segmentation_logits,
                             target_segmentation,
                             target_depth,
                             target_edge,
                             input_rgb,
                             depth_threshold=0.1,
                             edge_threshold=0.5,
                             black_threshold=0.1,
                             weights=(1.0, 0.1, 0.1, 0.1)):
    """
    Combines:
    - segmentation loss
    - smoothness consistency loss
    - discontinuity class transition loss
    - black near edge constraint
    """
 
    seg_weight, smoothness_weight, discontinuity_weight, black_weight = weights
 
    # --- Standard segmentation loss ---
    loss_segmentation = F.cross_entropy(pred_segmentation_logits, target_segmentation)
 
    # --- Predicted class map ---
    pred_classes = pred_segmentation_logits.argmax(dim=1)  # (B, H, W)
 
    # --- Neighbor differences ---
    def neighbor_diff(tensor):
        diff_x = (tensor[:, :, :, :-1] != tensor[:, :, :, 1:]).float()
        diff_y = (tensor[:, :, :-1, :] != tensor[:, :, 1:, :])
        return diff_x, diff_y
 
    pred_classes_ = pred_classes.unsqueeze(1).float()  # (B, 1, H, W) for broadcasting
    diff_x, diff_y = neighbor_diff(pred_classes_)
 
    # --- Depth/Edge discontinuities ---
    def gradient_x(img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]
 
    def gradient_y(img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]
 
    depth_grad_x = gradient_x(target_depth)
    depth_grad_y = gradient_y(target_depth)
 
    depth_jump_x = (depth_grad_x.abs() > depth_threshold)
    depth_jump_y = (depth_grad_y.abs() > depth_threshold)
 
    edge_jump_x = (target_edge[:, :, :, :-1] > edge_threshold)
    edge_jump_y = (target_edge[:, :, :-1, :] > edge_threshold)
 
    discontinuity_x = depth_jump_x | edge_jump_x
    discontinuity_y = depth_jump_y | edge_jump_y
 
    # Invert to find smooth regions
    smooth_region_x = (~discontinuity_x).float()
    smooth_region_y = (~discontinuity_y).float()
 
    # --- Loss for smooth regions: want NO class changes ---
    loss_smooth_x = (diff_x * smooth_region_x).mean()
    loss_smooth_y = (diff_y * smooth_region_y).mean()
    loss_smoothness = loss_smooth_x + loss_smooth_y
 
    # --- Loss for discontinuities: want class changes ---
    loss_disc_x = ((1 - diff_x) * discontinuity_x.float()).mean()
    loss_disc_y = ((1 - diff_y) * discontinuity_y.float()).mean()
    loss_discontinuity = loss_disc_x + loss_disc_y
 
    # --- Black near edge constraint ---
    brightness = input_rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)
 
    edge_mask = (target_edge > edge_threshold)  # (B, 1, H, W)
 
    # Expand edge mask to neighbors using 3x3 dilation
    kernel = torch.ones((1, 1, 3, 3), device=pred_segmentation_logits.device)
    neighbor_mask = F.conv2d(edge_mask.float(), kernel, padding=1)
    neighbor_mask = (neighbor_mask > 0)
 
    black_pixels = (brightness < black_threshold)
    black_near_edge = black_pixels & neighbor_mask
 
    pred_classes_expanded = pred_classes.unsqueeze(1)
    incorrect_black_class = (pred_classes_expanded != 0).float()
    loss_black = (incorrect_black_class * black_near_edge.float()).mean()
 
    # --- Total Loss ---
    total_loss = (seg_weight * loss_segmentation +
                  smoothness_weight * loss_smoothness +
                  discontinuity_weight * loss_discontinuity +
                  black_weight * loss_black)
 
    return total_loss, {
        "loss_segmentation": loss_segmentation.item(),
        "loss_smoothness": loss_smoothness.item(),
        "loss_discontinuity": loss_discontinuity.item(),
        "loss_black": loss_black.item()
    }


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
