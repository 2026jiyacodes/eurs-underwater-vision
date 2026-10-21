
import torch
import torch.nn as nn
from transformers import AutoModel


MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DINOv3Backbone(nn.Module):
    def __init__(self, model_id=MODEL_ID, freeze=True):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_id)

        self.hidden_size = self.encoder.config.hidden_size
        self.patch_size = self.encoder.config.patch_size
        self.num_register_tokens = getattr(
            self.encoder.config, "num_register_tokens", 0
        )
        self.freeze = freeze

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)

        # Keep the frozen feature extractor in evaluation mode.
        if self.freeze:
            self.encoder.eval()

        return self

    def forward(self, pixel_values):
        height, width = pixel_values.shape[-2:]

        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(pixel_values=pixel_values)
        else:
            outputs = self.encoder(pixel_values=pixel_values)

        # Remove the CLS token and register tokens.
        tokens = outputs.last_hidden_state
        patch_tokens = tokens[
            :, 1 + self.num_register_tokens :, :
        ]

        # Convert the patch-token sequence into a 2D feature map.
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        expected_tokens = grid_h * grid_w

        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens, "
                f"but received {patch_tokens.shape[1]}. "
                "Check the image size and patch size."
            )

        batch_size = pixel_values.shape[0]

        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            self.hidden_size,
            grid_h,
            grid_w,
        )

        return feature_map

