
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.restoration_model import UnderwaterRestorationNet

TRAIN_ROOT = Path(r"C:\Users\student\Downloads\eurs_train\train")
SCENE = "fish11"
IMAGE_NAME = "fish11_10_90.png"
IMAGE_SIZE = (512, 384)


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def metrics(prediction, target):
    mse = np.mean((prediction - target) ** 2)
    mae = np.mean(np.abs(prediction - target))
    psnr = float("inf") if mse == 0 else -10 * np.log10(mse)
    return psnr, mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "restoration_best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    model = UnderwaterRestorationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    degraded_path = TRAIN_ROOT / "degraded" / SCENE / IMAGE_NAME
    clean_path = TRAIN_ROOT / "gt_images" / f"{SCENE}.png"

    if not degraded_path.exists() or not clean_path.exists():
        raise FileNotFoundError(
            f"Missing image. Degraded: {degraded_path.exists()}, "
            f"clean: {clean_path.exists()}"
        )

    degraded = load_image(degraded_path)
    clean = load_image(clean_path)

    tensor = torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        restored = model(tensor.to(device))[0].cpu()
    restored = restored.permute(1, 2, 0).numpy()

    input_psnr, input_mae = metrics(degraded, clean)
    output_psnr, output_mae = metrics(restored, clean)

    print(f"Scene: {SCENE}")
    print(f"Image: {IMAGE_NAME}")
    print(f"Input PSNR:    {input_psnr:.2f} dB")
    print(f"Restored PSNR: {output_psnr:.2f} dB")
    print(f"Input MAE:     {input_mae:.5f}")
    print(f"Restored MAE:  {output_mae:.5f}")
    print("\nMean RGB values (0-255):")
    for name, image in [
        ("Degraded", degraded),
        ("Restored", restored),
        ("Clean", clean),
    ]:
        print(f"{name:10s}: {np.round(image.mean(axis=(0, 1)) * 255, 1)}")

    output_dir = PROJECT_ROOT / "predictions" / "restoration_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, image in [
        ("degraded", degraded),
        ("restored", restored),
        ("clean", clean),
    ]:
        Image.fromarray(
            np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
        ).save(output_dir / f"{SCENE}_{name}.png")

    print(f"\nImages saved to: {output_dir}")


if __name__ == "__main__":
    main()
