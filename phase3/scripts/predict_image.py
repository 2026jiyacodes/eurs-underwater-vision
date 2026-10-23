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

NUM_CLASSES = 5
INPUT_SIZE = (512, 384)  # width, height

CHECKPOINTS = {
    "best_epoch1": ROOT / "checkpoints" / "weighted_best.pt",
    "latest_epoch10": ROOT / "checkpoints" / "weighted_latest.pt",
}

CLASS_NAMES = [
    "background",
    "coral",
    "fish",
    "elasmobranch",
    "other",
]

PALETTE = np.array([
    [0, 0, 0],       # Background
    [255, 0, 0],     # Coral
    [0, 255, 0],     # Fish
    [0, 100, 255],   # Elasmobranch
    [255, 255, 0],   # Other
], dtype=np.uint8)


def main():
    if len(sys.argv) < 2:
        print('Usage: py scripts/predict_image.py "<image_path>"')
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    for name, path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} checkpoint: {path}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = EURSSegmenter(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    ).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    resized = image.resize(INPUT_SIZE, Image.Resampling.BICUBIC)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.asarray(
        processor.image_mean, dtype=np.float32
    ).reshape(1, 1, 3)

    std = np.asarray(
        processor.image_std, dtype=np.float32
    ).reshape(1, 1, 3)

    image_array = (image_array - mean) / std

    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(device)

    output_root = ROOT / "predictions"
    results = {}

    for name, checkpoint_path in CHECKPOINTS.items():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model.decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        with torch.inference_mode():
            logits = model(image_tensor)
            prediction = (
                logits.argmax(dim=1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        mask = Image.fromarray(prediction).resize(
            original_size, Image.Resampling.NEAREST
        )

        mask_array = np.asarray(mask)
        color_mask = Image.fromarray(PALETTE[mask_array])
        overlay = Image.blend(image, color_mask, alpha=0.45)

        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = output_dir / f"{image_path.stem}_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"

        mask.save(mask_path)
        overlay.save(overlay_path)

        results[name] = overlay

        print(f"\n--- {name.upper()} ---")
        print("Checkpoint epoch:", checkpoint.get("epoch"))
        print("Checkpoint validation mIoU:", checkpoint.get("val_miou"))
        print("Predicted classes:")

        for class_id in np.unique(mask_array):
            print(f"  {class_id}: {CLASS_NAMES[int(class_id)]}")

        print("Saved mask:", mask_path)
        print("Saved overlay:", overlay_path)

    # Create a single image for easy visual comparison.
    comparison = Image.new(
        "RGB",
        (original_size[0] * 3, original_size[1]),
    )
    comparison.paste(image, (0, 0))
    comparison.paste(results["best_epoch1"], (original_size[0], 0))
    comparison.paste(results["latest_epoch10"], (original_size[0] * 2, 0))

    comparison_path = output_root / f"{image_path.stem}_comparison.png"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(comparison_path)

    print("\nComparison saved:", comparison_path)
    print("Panels from left to right: Original | best (epoch 1) | latest (epoch 10)")


if __name__ == "__main__":
    main()