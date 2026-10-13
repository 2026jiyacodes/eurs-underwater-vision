from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

# Make the repository root importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoImageProcessor
from models.segmentation_model import EURSSegmenter
from models.dinov3_backbone import MODEL_ID


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_image.py <image_path>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load the pretrained backbone's image processor and model.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = EURSSegmenter(
        model_id=MODEL_ID,
        num_classes=5,
        freeze_backbone=True,
    ).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        logits = model(pixel_values)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Resize the predicted class map to the original image dimensions.
    mask = Image.fromarray(prediction, mode="L")
    mask = mask.resize(image.size, Image.Resampling.NEAREST)

    # Five illustrative colors, one for each class ID (0-4).
    palette = np.array([
        [0, 0, 0],        # Class 0: black
        [255, 0, 0],      # Class 1: red
        [0, 255, 0],      # Class 2: green
        [0, 100, 255],    # Class 3: blue
        [255, 255, 0],    # Class 4: yellow
    ], dtype=np.uint8)

    mask_array = np.asarray(mask)
    color_mask = Image.fromarray(palette[mask_array], mode="RGB")
    overlay = Image.blend(image, color_mask, alpha=0.45)

    output_dir = ROOT / "predictions"
    output_dir.mkdir(exist_ok=True)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"

    mask.save(mask_path)
    overlay.save(overlay_path)

    print("Input image:", image.size)
    print("Logits shape:", tuple(logits.shape))
    print("Predicted classes:", np.unique(mask_array).tolist())
    print("Saved mask:", mask_path)
    print("Saved overlay:", overlay_path)


if __name__ == "__main__":
    main()