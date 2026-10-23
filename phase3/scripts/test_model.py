
import torch
from PIL import Image
from transformers import AutoImageProcessor

from models.segmentation_model import EURSSegmenter
from models.dinov3_backbone import MODEL_ID


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)

model = EURSSegmenter(
    model_id=MODEL_ID,
    num_classes=5,
    freeze_backbone=True,
).to(device)

model.eval()

# Synthetic image for a forward-pass test only.
image = Image.new("RGB", (512, 512), color=(40, 100, 140))
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

with torch.no_grad():
    logits = model(pixel_values)
    predicted_mask = logits.argmax(dim=1)

print("Input:", tuple(pixel_values.shape))
print("Logits:", tuple(logits.shape))
print("Predicted mask:", tuple(predicted_mask.shape))
print("Classes present in this test:", torch.unique(predicted_mask).tolist())
print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ),
)

