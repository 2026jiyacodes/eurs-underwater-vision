
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

        # Select representations from transformer blocks 3, 6, 9 and 12.
        self.feature_indices = (3, 6, 9, 12)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)

        if self.freeze:
            self.encoder.eval()

        return self

    def forward(self, pixel_values):
        height, width = pixel_values.shape[-2:]

        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
        else:
            outputs = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        expected_tokens = grid_h * grid_w

        batch_size = pixel_values.shape[0]
        feature_maps = []

        for index in self.feature_indices:
            tokens = outputs.hidden_states[index]

            # Remove CLS and register tokens.
            patch_tokens = tokens[
                :, 1 + self.num_register_tokens :, :
            ]

            if patch_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"Layer {index}: expected {expected_tokens} "
                    f"patch tokens, got {patch_tokens.shape[1]}."
                )

            feature_map = patch_tokens.transpose(1, 2).reshape(
                batch_size,
                self.hidden_size,
                grid_h,
                grid_w,
            )

            feature_maps.append(feature_map)

        return feature_maps
