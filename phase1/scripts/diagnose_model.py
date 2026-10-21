
from pathlib import Path
import sys

# Add the project root so Python can import the models package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor
from tqdm import tqdm

from models.segmentation_model import EURSSegmenter


# ============================================================
# Configuration
# ============================================================

TRAIN_ROOT = Path(
    r"C:\Users\student\Downloads\eurs_train\train"
)

CHECKPOINT_PATH = ROOT / "checkpoints" / "weighted_best.pt"

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

NUM_CLASSES = 5

# (width, height), matching the training configuration
INPUT_SIZE = (512, 384)

VAL_SCENES = {"ray1", "seahorse1", "fish11", "fish12"}

CLASS_NAMES = [
    "background",
    "coral",
    "fish",
    "elasmobranch",
    "other",
]


# ============================================================
# Helpers
# ============================================================

def load_rgb_image(path):
    """Load an image and resize it exactly as in training."""
    image = Image.open(path).convert("RGB")
    image = image.resize(INPUT_SIZE, Image.Resampling.BICUBIC)
    return image


def load_mask(path):
    """Load a ground-truth mask using nearest-neighbor resizing."""
    mask = Image.open(path).convert("L")
    mask = mask.resize(INPUT_SIZE, Image.Resampling.NEAREST)

    mask = np.asarray(mask, dtype=np.int64)

    if mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(
            f"Unexpected label values in {path}: "
            f"min={mask.min()}, max={mask.max()}. "
            f"Expected labels 0 through {NUM_CLASSES - 1}."
        )

    return mask


def image_to_tensor(image, mean, std):
    """Convert an RGB PIL image into a normalized BCHW tensor."""
    image = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1)).copy()

    return torch.from_numpy(image).unsqueeze(0)


def update_confusion_matrix(confusion, targets, predictions):
    """
    Rows = ground-truth classes.
    Columns = predicted classes.
    """
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    valid = (
        (targets >= 0)
        & (targets < NUM_CLASSES)
        & (predictions >= 0)
        & (predictions < NUM_CLASSES)
    )

    encoded = NUM_CLASSES * targets[valid] + predictions[valid]

    confusion += np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion


def metrics_from_confusion(confusion):
    """Calculate segmentation metrics from a confusion matrix."""
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth_count = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)

    false_positive = predicted_count - true_positive
    false_negative = ground_truth_count - true_positive

    union = true_positive + false_positive + false_negative

    iou = np.divide(
        true_positive,
        union,
        out=np.full(NUM_CLASSES, np.nan),
        where=union > 0,
    )

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.full(NUM_CLASSES, np.nan),
        where=predicted_count > 0,
    )

    recall = np.divide(
        true_positive,
        ground_truth_count,
        out=np.full(NUM_CLASSES, np.nan),
        where=ground_truth_count > 0,
    )

    f1_denominator = 2 * true_positive + false_positive + false_negative

    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.full(NUM_CLASSES, np.nan),
        where=f1_denominator > 0,
    )

    total_pixels = confusion.sum()
    pixel_accuracy = (
        true_positive.sum() / total_pixels
        if total_pixels > 0
        else float("nan")
    )

    # Mean over classes that actually occur in ground truth or predictions.
    present_classes = union > 0
    miou = (
        float(np.mean(iou[present_classes]))
        if present_classes.any()
        else float("nan")
    )

    # Macro F1 over the same set of present classes.
    macro_f1 = (
        float(np.mean(f1[present_classes]))
        if present_classes.any()
        else float("nan")
    )

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_accuracy": pixel_accuracy,
        "miou": miou,
        "macro_f1": macro_f1,
        "present_classes": present_classes,
    }


# ============================================================
# Main evaluation
# ============================================================

def main():
    print("=" * 72)
    print("EURS SEGMENTATION MODEL DIAGNOSTICS")
    print("=" * 72)

    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_ROOT}\n"
            "Update TRAIN_ROOT at the top of this script."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Make sure seg_baseline_best.pt exists in checkpoints/."
        )

    degraded_root = TRAIN_ROOT / "degraded"
    masks_root = TRAIN_ROOT / "gt_masks"

    if not degraded_root.exists():
        raise FileNotFoundError(f"Missing directory: {degraded_root}")

    if not masks_root.exists():
        raise FileNotFoundError(f"Missing directory: {masks_root}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Match the image normalization used during training.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = processor.image_mean
    std = processor.image_std

    # Reconstruct the same architecture used for the checkpoint.
    model = EURSSegmenter(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if "decoder_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'decoder_state_dict'. "
            "This script expects the checkpoint format created by train.py."
        )

    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.to(device)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    if "val_miou" in checkpoint:
        print(f"Saved validation mIoU: {checkpoint['val_miou']:.4f}")

    # Discover every image belonging to the fixed validation scenes.
    samples = []
    scene_counts = {}

    for scene in sorted(VAL_SCENES):
        scene_dir = degraded_root / scene
        mask_path = masks_root / f"{scene}.png"

        if not scene_dir.is_dir():
            raise FileNotFoundError(
                f"Validation scene directory not found: {scene_dir}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth mask not found: {mask_path}"
            )

        image_paths = sorted(
            path
            for path in scene_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        if not image_paths:
            raise RuntimeError(f"No images found in {scene_dir}")

        scene_counts[scene] = len(image_paths)

        for image_path in image_paths:
            samples.append((scene, image_path, mask_path))

    print("\nValidation scenes:")
    for scene, count in scene_counts.items():
        print(f"  {scene}: {count} images")

    print(f"Total images: {len(samples)}")

    global_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    scene_confusions = {
        scene: np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )
        for scene in VAL_SCENES
    }

    failed = []

    with torch.inference_mode():
        for scene, image_path, mask_path in tqdm(
            samples,
            desc="Evaluating",
            unit="image",
        ):
            try:
                image = load_rgb_image(image_path)
                target = load_mask(mask_path)

                input_tensor = image_to_tensor(
                    image,
                    mean,
                    std,
                ).to(device)

                logits = model(input_tensor)

                if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
                    raise RuntimeError(
                        f"Unexpected model output shape: {tuple(logits.shape)}"
                    )

                prediction = (
                    logits.argmax(dim=1)[0]
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )

                if prediction.shape != target.shape:
                    raise RuntimeError(
                        f"Prediction shape {prediction.shape} "
                        f"does not match target shape {target.shape}"
                    )

                update_confusion_matrix(
                    global_confusion,
                    target,
                    prediction,
                )

                update_confusion_matrix(
                    scene_confusions[scene],
                    target,
                    prediction,
                )

            except Exception as exc:
                failed.append((str(image_path), str(exc)))

    if failed:
        print(f"\nWARNING: {len(failed)} image(s) failed evaluation.")
        for path, error in failed[:10]:
            print(f"  {path}\n    {error}")

    successful_count = len(samples) - len(failed)

    if successful_count == 0:
        raise RuntimeError("No validation images were evaluated successfully.")

    # --------------------------------------------------------
    # Overall segmentation metrics
    # --------------------------------------------------------

    results = metrics_from_confusion(global_confusion)

    print("\n" + "=" * 72)
    print("OVERALL VALIDATION METRICS")
    print("=" * 72)

    print(f"Successfully evaluated: {successful_count}/{len(samples)}")
    print(f"Pixel accuracy:         {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU:               {results['miou']:.4f}")
    print(f"Macro F1:               {results['macro_f1']:.4f}")

    print("\nPer-class metrics:")
    print(
        f"{'Class':<18}"
        f"{'IoU':>10}"
        f"{'F1':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'GT pixels':>14}"
        f"{'Pred pixels':>14}"
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        iou = results["iou"][class_id]
        f1 = results["f1"][class_id]
        precision = results["precision"][class_id]
        recall = results["recall"][class_id]

        print(
            f"{class_name:<18}"
            f"{iou:>10.4f}"
            f"{f1:>10.4f}"
            f"{precision:>12.4f}"
            f"{recall:>10.4f}"
            f"{global_confusion[class_id, :].sum():>14,d}"
            f"{global_confusion[:, class_id].sum():>14,d}"
        )

    # --------------------------------------------------------
    # Per-scene diagnostics
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("PER-SCENE METRICS")
    print("=" * 72)

    print(
        f"{'Scene':<18}"
        f"{'Images':>10}"
        f"{'Pixel Acc.':>14}"
        f"{'mIoU':>12}"
        f"{'Macro F1':>12}"
    )

    for scene in sorted(VAL_SCENES):
        scene_results = metrics_from_confusion(scene_confusions[scene])

        print(
            f"{scene:<18}"
            f"{scene_counts[scene]:>10,d}"
            f"{scene_results['pixel_accuracy']:>14.4f}"
            f"{scene_results['miou']:>12.4f}"
            f"{scene_results['macro_f1']:>12.4f}"
        )

    # --------------------------------------------------------
    # Qseg diagnostic
    # --------------------------------------------------------

    # F1 and mIoU are calculated above. 
    provisional_qseg = results["macro_f1"] * results["miou"]

    print("\n" + "=" * 72)
    print("Qseg DIAGNOSTIC")
    print("=" * 72)

    print(f"F1 used here (macro):    {results['macro_f1']:.4f}")
    print(f"mIoU:                    {results['miou']:.4f}")
    print(f"F1 x mIoU:               {provisional_qseg:.4f}")
    print(
        "NOTE: F1 x mIoU is a provisional diagnostic, NOT the official "
        "Qseg, because the official class-coverage penalty has not been "
        "verified against the evaluator implementation."
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
