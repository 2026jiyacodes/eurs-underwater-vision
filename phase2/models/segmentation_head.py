
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 5


class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes=NUM_CLASSES):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels, 256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, num_classes, kernel_size=1
            ),
        )

    def forward(self, feature_map, output_size):
        logits = self.head(feature_map)

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

