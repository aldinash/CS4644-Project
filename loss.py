import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEdgeCouplingLoss(nn.Module):
    """
    Depth-Edge Coupling Loss:
    Penalizes depth gradients at non-edge pixels.
    Inputs:
        - depth: (B, 1, H, W) predicted depth map
        - edge: (B, 1, H, W) predicted binary edge mask (0 or 1)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Fixed 3x3 Sobel kernels for finite grads
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()

        self.register_buffer('kx', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('ky', sobel_y.view(1, 1, 3, 3))

    def forward(self, depth: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) predicted depth map
            edge: (B, 1, H, W) predicted binary edge map (0 or 1)
        Returns:
            scalar loss (torch.Tensor)
        """
        gx = F.conv2d(depth, self.kx, padding=1, padding_mode='reflect')
        gy = F.conv2d(depth, self.ky, padding=1, padding_mode='reflect')
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

        weight = (1.0 - edge)  # 1 at egde pixels


        loss = (weight * grad_mag).mean()
        return loss
