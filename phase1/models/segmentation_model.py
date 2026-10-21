
import torch.nn as nn

from models.dinov3_backbone import DINOv3Backbone
from models.segmentation_head import SimpleSegmentationDecoder


class EURSSegmenter(nn.Module):
    def __init__(self, model_id=None, num_classes=5, freeze_backbone=True):
        super().__init__()

        if model_id is None:
            self.backbone = DINOv3Backbone(
                freeze=freeze_backbone
            )
        else:
            self.backbone = DINOv3Backbone(
                model_id=model_id,
                freeze=freeze_backbone,
            )

        self.decoder = SimpleSegmentationDecoder(
            in_channels=self.backbone.hidden_size,
            num_classes=num_classes,
        )

    def forward(self, pixel_values):
        feature_map = self.backbone(pixel_values)

        logits = self.decoder(
            feature_map,
            output_size=pixel_values.shape[-2:],
        )

        return logits

