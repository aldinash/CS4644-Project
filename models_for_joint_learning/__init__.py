from .unet_adaptive_bins import UnetAdaptiveBins

self.edge_head = nn.Sequential(
    nn.Conv2d(128, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 1, kernel_size=1)   # Single channel for edge prediction
)
