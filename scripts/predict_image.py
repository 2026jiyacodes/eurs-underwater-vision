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

CHECKPOINT_PATH = ROOT / "checkpoints" / "seg_baseline_best.pt"

CLASS_NAMES = [
    "background",
    "coral",
    "fish",
    "elasmobranch",
    "other",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: py scripts/predict_image.py <image_path>")
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = EURSSegmenter(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    ).to(device)

    model.decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    resized = image.resize(
        INPUT_SIZE,
        Image.Resampling.BICUBIC,
    )
    image_array = np.asarray(
        resized, dtype=np.float32
    ) / 255.0

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

    with torch.inference_mode():
        logits = model(image_tensor)
        prediction = (
            logits.argmax(dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    # Resize the label map using nearest-neighbor interpolation.
    mask = Image.fromarray(prediction)
    mask = mask.resize(
        original_size,
        Image.Resampling.NEAREST,
    )

    mask_array = np.asarray(mask)

    # Colors are for visualization only.
    palette = np.array([
        [0, 0, 0],        # Background
        [255, 0, 0],      # Coral
        [0, 255, 0],      # Fish
        [0, 100, 255],    # Elasmobranch
        [255, 255, 0],    # Other
    ], dtype=np.uint8)

    color_mask = Image.fromarray(palette[mask_array])
    overlay = Image.blend(image, color_mask, alpha=0.45)

    output_dir = ROOT / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"

    # Save the actual segmentation labels as a single-channel PNG.
    mask.save(mask_path)
    overlay.save(overlay_path)

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint validation mIoU:", checkpoint.get("val_miou"))
    print("Original image size:", original_size)
    print("Logits shape:", tuple(logits.shape))
    print("Predicted classes:")

    for class_id in np.unique(mask_array):
        print(f"  {class_id}: {CLASS_NAMES[int(class_id)]}")

    print("Saved mask:", mask_path)
    print("Saved overlay:", overlay_path)


if __name__ == "__main__":
    main()