
import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.restoration_model import UnderwaterRestorationNet


VAL_SCENES = {"ray1", "seahorse1", "fish11", "fish12"}
IMAGE_SIZE = (512, 384)  # width, height


def collect_pairs(train_root):
    """Pair every degraded image with its scene's clean reference."""
    train_root = Path(train_root)
    degraded_root = train_root / "degraded"
    clean_root = train_root / "gt_images"

    if not degraded_root.is_dir() or not clean_root.is_dir():
        raise FileNotFoundError(
            f"Expected degraded/ and gt_images/ inside {train_root}"
        )

    pairs = []

    for scene_dir in sorted(degraded_root.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene = scene_dir.name
        clean_path = clean_root / f"{scene}.png"

        if not clean_path.is_file():
            print(f"Skipping {scene}: clean target not found")
            continue

        for degraded_path in sorted(scene_dir.glob("*.png")):
            pairs.append((degraded_path, clean_path, scene))

    if not pairs:
        raise RuntimeError("No paired degraded/clean PNG images found.")

    return pairs


def limit_pairs(pairs, maximum):
    """Select samples from across scenes."""
    if maximum <= 0 or len(pairs) <= maximum:
        return pairs

    by_scene = {}
    for pair in pairs:
        by_scene.setdefault(pair[2], []).append(pair)

    selected = []
    scenes = sorted(by_scene)
    per_scene = maximum // len(scenes)
    remainder = maximum % len(scenes)

    for index, scene in enumerate(scenes):
        scene_pairs = by_scene[scene]
        count = per_scene + (1 if index < remainder else 0)

        if count == 0:
            continue

        indices = np.linspace(
            0,
            len(scene_pairs) - 1,
            num=min(count, len(scene_pairs)),
            dtype=int,
        )
        selected.extend(scene_pairs[i] for i in indices)

    return selected


class RestorationDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    @staticmethod
    def load_image(path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)

        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        degraded_path, clean_path, _ = self.pairs[index]

        degraded = self.load_image(degraded_path)
        clean = self.load_image(clean_path)

        # Apply the same spatial transforms to input and target.
        if self.augment and random.random() < 0.5:
            degraded = degraded.flip(-1)
            clean = clean.flip(-1)

        if self.augment and random.random() < 0.5:
            degraded = degraded.flip(-2)
            clean = clean.flip(-2)

        return degraded, clean


def psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2).item()

    if mse == 0:
        return float("inf")

    return -10.0 * np.log10(mse)


def evaluate(model, loader, device):
    model.eval()
    loss_function = nn.L1Loss()

    total_loss = 0.0
    input_psnr_sum = 0.0
    restored_psnr_sum = 0.0
    count = 0

    with torch.no_grad():
        for degraded, clean in loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            restored = model(degraded)

            total_loss += loss_function(restored, clean).item()
            input_psnr_sum += psnr(degraded, clean)
            restored_psnr_sum += psnr(restored, clean)
            count += 1

    if count == 0:
        raise RuntimeError("Validation loader is empty.")

    return (
        total_loss / count,
        input_psnr_sum / count,
        restored_psnr_sum / count,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=800)
    parser.add_argument("--max-val-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    pairs = collect_pairs(args.train_root)

    train_pairs = [p for p in pairs if p[2] not in VAL_SCENES]
    val_pairs = [p for p in pairs if p[2] in VAL_SCENES]

    train_pairs = limit_pairs(train_pairs, args.max_train_samples)
    val_pairs = limit_pairs(val_pairs, args.max_val_samples)

    if not train_pairs or not val_pairs:
        raise RuntimeError(
            "Training or validation split is empty. "
            "Check VAL_SCENES against your dataset."
        )

    train_scenes = sorted({p[2] for p in train_pairs})
    val_scenes = sorted({p[2] for p in val_pairs})

    if set(train_scenes) & set(val_scenes):
        raise RuntimeError("Scene leakage detected between train and validation.")

    print("Training pairs:", len(train_pairs))
    print("Validation pairs:", len(val_pairs))
    print("Training scenes:", train_scenes)
    print("Validation scenes:", val_scenes)

    train_loader = DataLoader(
        RestorationDataset(train_pairs, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        RestorationDataset(val_pairs, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UnderwaterRestorationNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    loss_function = nn.L1Loss()

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "restoration_best.pt"

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, (degraded, clean) in enumerate(train_loader, start=1):
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad(set_to_none=True)

            restored = model(degraded)
            loss = loss_function(restored, clean)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Step {step}/{len(train_loader)} "
                    f"Train L1: {running_loss / step:.5f}"
                )

        val_loss, input_psnr, restored_psnr = evaluate(
            model, val_loader, device
        )

        print(
            f"\nEpoch {epoch + 1}: "
            f"Val L1={val_loss:.5f}, "
            f"Input PSNR={input_psnr:.2f} dB, "
            f"Restored PSNR={restored_psnr:.2f} dB\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_l1": val_loss,
                    "input_psnr": input_psnr,
                    "restored_psnr": restored_psnr,
                    "image_size": IMAGE_SIZE,
                },
                checkpoint_path,
            )

            print(f"Saved best restoration checkpoint: {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
