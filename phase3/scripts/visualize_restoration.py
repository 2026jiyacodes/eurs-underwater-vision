
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.restoration_model import UnderwaterRestorationNet


IMAGE_SIZE = (512, 384)  # width, height
VAL_SCENES = ["ray1", "seahorse1", "fish11", "fish12"]


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return image, tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "restoration_best.pt"),
    )
    args = parser.parse_args()

    train_root = Path(args.train_root)
    output_dir = PROJECT_ROOT / "predictions" / "restoration_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = UnderwaterRestorationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        for scene in VAL_SCENES:
            degraded_dir = train_root / "degraded" / scene
            clean_path = train_root / "gt_images" / f"{scene}.png"

            if not degraded_dir.is_dir():
                print(f"Skipping {scene}: degraded folder not found")
                continue

            if not clean_path.is_file():
                print(f"Skipping {scene}: clean reference not found")
                continue

            candidates = sorted(degraded_dir.glob("*.png"))

            if not candidates:
                print(f"Skipping {scene}: no degraded PNG images")
                continue

            # Use one representative degraded image per validation scene.
            degraded_path = candidates[0]

            degraded_image, input_tensor = load_image(degraded_path)
            clean_image = Image.open(clean_path).convert("RGB")
            clean_image = clean_image.resize(
                IMAGE_SIZE, Image.Resampling.BICUBIC
            )

            input_tensor = input_tensor.to(device)
            restored_tensor = model(input_tensor)

            restored_array = (
                restored_tensor[0]
                .clamp(0, 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            restored_image = Image.fromarray(
                (restored_array * 255).round().astype(np.uint8)
            )

            # Three panels: degraded | restored | clean reference.
            label_height = 36
            comparison = Image.new(
                "RGB",
                (IMAGE_SIZE[0] * 3, IMAGE_SIZE[1] + label_height),
                "white",
            )

            panels = [
                ("DEGRADED", degraded_image),
                ("RESTORED", restored_image),
                ("CLEAN REFERENCE", clean_image),
            ]

            draw = ImageDraw.Draw(comparison)

            for index, (label, image) in enumerate(panels):
                x = index * IMAGE_SIZE[0]
                comparison.paste(image, (x, label_height))
                draw.text((x + 10, 10), label, fill="black")

            output_path = output_dir / f"{scene}_comparison.png"
            comparison.save(output_path)

            print(f"Saved: {output_path}")
            print(f"Input image: {degraded_path.name}")

    print(f"\nAll comparisons saved in: {output_dir}")


if __name__ == "__main__":
    main()
