from pathlib import Path
import argparse
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segmentation_model import EURSSegmenter
from models.dinov3_backbone import MODEL_ID


TRAIN_ROOT = Path(r"<training dataset path")
NUM_CLASSES = 5
CLASS_WEIGHTS = [1.0, 1.0, 2.0, 4.0, 2.5]
VAL_SCENES = {"ray1", "seahorse1", "fish11", "fish12"}

INPUT_SIZE = (512, 384)  # width, height


class EURSDataset(Dataset):
    def __init__(self, samples, mean, std):
        self.samples = samples
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as im:
            image = im.convert("RGB").resize(
                INPUT_SIZE, Image.Resampling.BICUBIC
            )
            image_array = np.array(image, dtype=np.float32) / 255.0

        with Image.open(mask_path) as im:
            mask = im.convert("L").resize(
                INPUT_SIZE, Image.Resampling.NEAREST
            )
            mask_array = np.array(mask, dtype=np.uint8)

        if mask_array.max() >= NUM_CLASSES:
            raise ValueError(
                f"Unexpected label in {mask_path}: "
                f"{np.unique(mask_array).tolist()}"
            )

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )
        image_tensor = (image_tensor - self.mean) / self.std

        mask_tensor = torch.from_numpy(mask_array.astype(np.int64).copy())

        return image_tensor, mask_tensor


def discover_scenes(train_root):
    degraded_root = train_root / "degraded"
    masks_root = train_root / "gt_masks"

    scenes = []

    for scene_dir in sorted(degraded_root.iterdir()):
        if not scene_dir.is_dir():
            continue

        mask_path = masks_root / f"{scene_dir.name}.png"

        if not mask_path.is_file():
            print(f"Skipping {scene_dir.name}: no matching mask")
            continue

        image_paths = sorted(
            p for p in scene_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        if image_paths:
            scenes.append((scene_dir.name, image_paths, mask_path))

    if len(scenes) < 2:
        raise RuntimeError(
            f"Expected at least two labeled scenes. Found {len(scenes)}."
        )

    return scenes


def build_samples(scenes):
    return [
        (image_path, mask_path)
        for _, image_paths, mask_path in scenes
        for image_path in image_paths
    ]


def confusion_matrix_update(matrix, targets, predictions):
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    valid = (targets >= 0) & (targets < NUM_CLASSES)
    encoded = NUM_CLASSES * targets[valid] + predictions[valid]

    matrix += torch.bincount(
        encoded, minlength=NUM_CLASSES ** 2
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_miou(matrix):
    intersection = matrix.diag()
    union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection

    present = union > 0
    if not present.any():
        return 0.0

    return (intersection[present].float() / union[present]).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-root",type=Path, default=TRAIN_ROOT, help="Path to training dataset directory")
    args = parser.parse_args()
    train_root = args.train_root

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    scenes = discover_scenes(train_root)
    scene_names = {scene[0] for scene in scenes}

    missing = VAL_SCENES - scene_names
    if missing:
        raise RuntimeError(
            f"Validation scenes not found: {sorted(missing)}"
        )

    val_scenes = [
        scene for scene in scenes
        if scene[0] in VAL_SCENES
    ]

    train_scenes = [
    scene for scene in scenes
        if scene[0] not in VAL_SCENES
    ]

    train_samples = build_samples(train_scenes)
    val_samples = build_samples(val_scenes)

    # Shuffle images across scenes before limiting the smoke test.
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    if args.max_train_samples is not None:
        train_samples = train_samples[:args.max_train_samples]
    if args.max_val_samples is not None:
        val_samples = val_samples[:args.max_val_samples]

    print("Training scenes:", [s[0] for s in train_scenes])
    print("Validation scenes:", [s[0] for s in val_scenes])
    print("Training images:", len(train_samples))
    print("Validation images:", len(val_samples))

    train_ds = EURSDataset(
        train_samples, processor.image_mean, processor.image_std
    )
    val_ds = EURSDataset(
        val_samples, processor.image_mean, processor.image_std
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = EURSSegmenter(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = ROOT / "checkpoints"
    output_dir.mkdir(exist_ok=True)

    best_miou = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            if step % 5 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Step {step}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        matrix = torch.zeros(
            NUM_CLASSES, NUM_CLASSES, dtype=torch.int64
        )

        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss_sum += loss.item()
                predictions = logits.argmax(dim=1)

                confusion_matrix_update(
                    matrix, masks.cpu(), predictions.cpu()
                )

        avg_val_loss = val_loss_sum / len(val_loader)
        miou = calculate_miou(matrix)

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_mIoU={miou:.4f}\n"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_id": MODEL_ID,
            "num_classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
            "decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": miou,
        }

        torch.save(checkpoint, output_dir / "weighted_latest.pt")

        if miou > best_miou:
            best_miou = miou
            torch.save(checkpoint, output_dir / "weighted_best.pt")
            print("Saved new best checkpoint.")

    print("Training finished.")
    print("Checkpoints:", output_dir)


if __name__ == "__main__":
    main()