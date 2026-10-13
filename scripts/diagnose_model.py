from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segmentation_model import EURSSegmenter
from models.dinov3_backbone import MODEL_ID

TRAIN_ROOT = Path(r"C:\Users\student\Downloads\eurs_train\train")
CHECKPOINT = ROOT / "checkpoints" / "best.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    model = EURSSegmenter(
        model_id=checkpoint["model_id"],
        num_classes=checkpoint["num_classes"],
        freeze_backbone=True,
    ).to(device)

    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    # Choose one labeled scene and one of its degraded images.
    image_path = (
        TRAIN_ROOT / "degraded" / "fish1" / "fish1_10_10.png"
    )
    mask_path = TRAIN_ROOT / "gt_masks" / "fish1.png"

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    with Image.open(mask_path) as im:
        target = np.array(im.convert("L").resize(
            image.size, Image.Resampling.NEAREST
        ))

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        logits = model(pixel_values)
        prediction = logits.argmax(dim=1)[0].cpu().numpy()

    # Resize predictions to original resolution for comparison.
    pred_image = Image.fromarray(
        prediction.astype(np.uint8), mode="L"
    ).resize(image.size, Image.Resampling.NEAREST)
    prediction = np.array(pred_image)

    print("Device:", device)
    print("Input size:", image.size)
    print("Ground-truth distribution:")
    print(dict(zip(*np.unique(target, return_counts=True))))
    print("Prediction distribution:")
    print(dict(zip(*np.unique(prediction, return_counts=True))))

    output_dir = ROOT / "predictions"
    output_dir.mkdir(exist_ok=True)

    pred_image.save(output_dir / "diagnostic_mask.png")

    print("Saved:", output_dir / "diagnostic_mask.png")


if __name__ == "__main__":
    main()