
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TRAIN_ROOT = Path(r"C:\Users\maheit\Downloads\eurs_train\train")

MASKS_DIR = TRAIN_ROOT / "gt_masks"
DEGRADED_DIR = TRAIN_ROOT / "degraded"

CLASS_NAMES = {
    0: "background",
    1: "coral",
    2: "fish",
    3: "elasmobranch",
    4: "other",
}

NUM_CLASSES = len(CLASS_NAMES)


def main():
    if not MASKS_DIR.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {MASKS_DIR}")

    # Each scene has one ground-truth mask shared by its degraded images.
    mask_paths = sorted(MASKS_DIR.glob("*.png"))

    if not mask_paths:
        raise RuntimeError(f"No PNG masks found in {MASKS_DIR}")

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    scene_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixels = 0

    print(f"Found {len(mask_paths)} ground-truth masks.")
    print("Scanning masks...")

    for mask_path in tqdm(mask_paths, desc="Masks"):
        mask = np.asarray(
            Image.open(mask_path).convert("L"),
            dtype=np.int64,
        )

        labels, counts = np.unique(mask, return_counts=True)

        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(
                f"Unexpected labels in {mask_path}: "
                f"{labels.tolist()}"
            )

        total_pixels += mask.size

        for label, count in zip(labels, counts):
            pixel_counts[label] += count
            image_counts[label] += 1
            scene_counts[label] += 1

    # Count degraded images associated with each class.
    degraded_images_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    scenes_without_class = defaultdict(list)

    for mask_path in mask_paths:
        scene = mask_path.stem
        mask = np.asarray(
            Image.open(mask_path).convert("L"),
            dtype=np.uint8,
        )

        present_labels = set(np.unique(mask).tolist())

        scene_dir = DEGRADED_DIR / scene
        if scene_dir.is_dir():
            n_images = sum(
                1 for p in scene_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
        else:
            n_images = 0

        for class_id in CLASS_NAMES:
            if class_id in present_labels:
                degraded_images_per_class[class_id] += n_images
            else:
                scenes_without_class[class_id].append(scene)

    print("\n" + "=" * 78)
    print("GROUND-TRUTH CLASS DISTRIBUTION")
    print("=" * 78)

    print(
        f"{'Class':<18}"
        f"{'Pixels':>16}"
        f"{'Pixel %':>12}"
        f"{'Scenes':>12}"
        f"{'Degraded imgs':>18}"
    )

    for class_id, class_name in CLASS_NAMES.items():
        percentage = (
            100.0 * pixel_counts[class_id] / total_pixels
            if total_pixels else 0.0
        )

        print(
            f"{class_name:<18}"
            f"{pixel_counts[class_id]:>16,d}"
            f"{percentage:>11.3f}%"
            f"{image_counts[class_id]:>12,d}"
            f"{degraded_images_per_class[class_id]:>18,d}"
        )

    print(f"\nTotal ground-truth pixels: {total_pixels:,}")

    print("\nScenes without each class:")
    for class_id, class_name in CLASS_NAMES.items():
        missing = scenes_without_class[class_id]
        print(
            f"  {class_name}: "
            f"{len(missing)} scenes"
            + (f" — {', '.join(missing)}" if missing else "")
        )

    print("\n" + "=" * 78)
    print("RARE-CLASS SUMMARY")
    print("=" * 78)

    for class_id in range(1, NUM_CLASSES):
        fraction = (
            pixel_counts[class_id] / total_pixels
            if total_pixels else 0.0
        )
        print(
            f"{CLASS_NAMES[class_id]}: "
            f"{fraction * 100:.3f}% of all pixels"
        )

    print("\nDataset diagnostic complete.")


if __name__ == "__main__":
    main()
