from pathlib import Path
import argparse
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks for the EURS Dev dataset."
    )
    parser.add_argument(
        "--dev-root",
        type=Path,
        required=True,
        help="Path to the Dev directory containing the degraded folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "seg_masks_raw",
        help="Directory for raw predicted masks.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process; 0 means process all images.",
    )
    args = parser.parse_args()

    degraded_root = args.dev_root / "degraded"

    if not degraded_root.is_dir():
        raise FileNotFoundError(
            f"Could not find the degraded directory: {degraded_root}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    image_paths = sorted(
        path
        for path in degraded_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No images found under {degraded_root}")

    if args.max_images < 0:
        raise ValueError("--max-images must be zero or a positive integer.")

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(processor.image_std, dtype=np.float32).reshape(1, 1, 3)

    model = EURSSegmenter(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    ).to(device)

    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint validation mIoU: {checkpoint.get('val_miou')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            resized = image.resize(INPUT_SIZE, Image.Resampling.BICUBIC)

            image_array = np.asarray(resized, dtype=np.float32) / 255.0
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

            # Preserve scene folders to avoid filename collisions.
            relative_path = image_path.relative_to(degraded_root)
            output_path = (
                args.output_dir / relative_path.parent / f"{image_path.stem}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save a single-channel mask containing class IDs 0–4.
            Image.fromarray(prediction, mode="L").save(output_path)
            successful += 1

            if index == 1 or index % 50 == 0 or index == len(image_paths):
                print(
                    f"Progress: {index}/{len(image_paths)} "
                    f"| Successful: {successful} "
                    f"| Latest: {relative_path}"
                )

        except Exception as exc:
            failed.append((str(image_path), str(exc)))
            print(f"ERROR processing {image_path}: {exc}")

    print("\n--- Inference summary ---")
    print(f"Images selected: {len(image_paths)}")
    print(f"Masks saved: {successful}")
    print(f"Failures: {len(failed)}")
    print(f"Output directory: {args.output_dir}")

    if failed:
        print("\nFailed images:")
        for path, error in failed:
            print(f"  {path}: {error}")

    if successful != len(image_paths):
        raise RuntimeError(
            "Inference was incomplete. Resolve the errors before submission."
        )


if __name__ == "__main__":
    main()