
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 5


class FPNDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=NUM_CLASSES,
        fpn_channels=128,
    ):
        super().__init__()

        # Project the four DINOv3 feature maps into a common
        # channel dimension.
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, fpn_channels, kernel_size=1)
            for _ in range(4)
        ])

        def smoothing_block():
            return nn.Sequential(
                nn.Conv2d(
                    fpn_channels,
                    fpn_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, fpn_channels),
                nn.GELU(),
            )

        self.smooth = nn.ModuleList([
            smoothing_block() for _ in range(4)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(
                fpn_channels, fpn_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(8, fpn_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

    def forward(self, feature_maps, output_size):
        # All four ViT features initially have the same patch-grid
        # resolution. Construct a pyramid at 4x, 2x, 1x and 1/2x.
        base_h, base_w = feature_maps[0].shape[-2:]

        sizes = [
            (base_h * 4, base_w * 4),
            (base_h * 2, base_w * 2),
            (base_h, base_w),
            (max(1, base_h // 2), max(1, base_w // 2)),
        ]

        resized = [
            F.interpolate(
                feature_maps[i],
                size=sizes[i],
                mode="bilinear",
                align_corners=False,
            )
            for i in range(4)
        ]

        lateral = [
            self.lateral_convs[i](resized[i])
            for i in range(4)
        ]

        # Top-down feature fusion.
        pyramid = [None] * 4
        pyramid[3] = lateral[3]

        for i in range(2, -1, -1):
            top_down = F.interpolate(
                pyramid[i + 1],
                size=lateral[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid[i] = lateral[i] + top_down

        pyramid = [
            self.smooth[i](pyramid[i])
            for i in range(4)
        ]

        # Use the finest pyramid level for pixel classification.
        logits = self.head(pyramid[0])

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
