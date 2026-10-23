from pathlib import Path
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segmentation_model import EURSSegmenter
from models.dinov3_backbone import MODEL_ID

NUM_CLASSES = 5
INPUT_SIZE = (512, 384)  # width, height, same as train.py
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_NAMES = ["background", "coral", "fish", "elasmobranch", "other"]
PALETTE = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 100, 255], [255, 255, 0]],
    dtype=np.uint8,
)


def load_model(ckpt_path, device):
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.asarray(processor.image_mean, np.float32).reshape(1, 1, 3)
    std = np.asarray(processor.image_std, np.float32).reshape(1, 1, 3)

    model = EURSSegmenter(
        model_id=MODEL_ID, num_classes=NUM_CLASSES, freeze_backbone=True
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path.name} (epoch {ckpt.get('epoch')}, val_mIoU {ckpt.get('val_miou')})")
    return model, mean, std


@torch.inference_mode()
def predict(model, image, mean, std, device, out_size=None):
    """Return a uint8 class map at out_size (width, height); default = image size."""
    out_size = out_size or image.size
    x = np.asarray(image.resize(INPUT_SIZE, Image.Resampling.BICUBIC), np.float32) / 255.0
    x = (x - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    logits = model(x)
    logits = F.interpolate(
        logits.float(), size=(out_size[1], out_size[0]),
        mode="bilinear", align_corners=False,
    )
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def titled(tile, title, header=22):
    canvas = Image.new("RGB", (tile.width, tile.height + header), (30, 30, 30))
    canvas.paste(tile, (0, header))
    ImageDraw.Draw(canvas).text((6, 5), title, fill=(255, 255, 255))
    return canvas


def legend_strip(width, height=26):
    strip = Image.new("RGB", (width, height), (30, 30, 30))
    d = ImageDraw.Draw(strip)
    x = 8
    for color, name in zip(PALETTE, CLASS_NAMES):
        d.rectangle([x, 7, x + 12, 19], fill=tuple(int(c) for c in color), outline=(255, 255, 255))
        d.text((x + 18, 7), name, fill=(255, 255, 255))
        x += 18 + 8 * len(name) + 18
    return strip


def make_panel(image, pred, gt=None):
    color = Image.fromarray(PALETTE[pred])
    tiles = [
        titled(image, "Original"),
        titled(color, "Predicted segmentation"),
        titled(Image.blend(image, color, 0.45), "Overlay"),
    ]
    if gt is not None:
        tiles.append(titled(Image.fromarray(PALETTE[gt]), "Ground truth"))
    w = sum(t.width for t in tiles)
    panel = Image.new("RGB", (w, tiles[0].height + 26))
    x = 0
    for t in tiles:
        panel.paste(t, (x, 0))
        x += t.width
    panel.paste(legend_strip(w), (0, tiles[0].height))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder containing degraded/ (and gt_masks/ if available)")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "weighted_best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "predictions" / "viz")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene names; default = all")
    ap.add_argument("--per-scene", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open each panel after saving")
    args = ap.parse_args()

    degraded = args.root / "degraded"
    if not degraded.is_dir():
        raise FileNotFoundError(f"No 'degraded' folder under {args.root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model(args.ckpt, device)
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    for scene_dir in sorted(p for p in degraded.iterdir() if p.is_dir()):
        if args.scenes and scene_dir.name not in args.scenes:
            continue
        imgs = sorted(p for p in scene_dir.rglob("*") if p.suffix.lower() in EXTS)
        if not imgs:
            continue

        gt_path = args.root / "gt_masks" / f"{scene_dir.name}.png"
        for path in random.sample(imgs, min(args.per_scene, len(imgs))):
            image = Image.open(path).convert("RGB")
            pred = predict(model, image, mean, std, device)

            gt = None
            if gt_path.is_file():
                gt = np.asarray(
                    Image.open(gt_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
                )

            panel = make_panel(image, pred, gt)
            out_path = args.out / f"{scene_dir.name}_{path.stem}.png"
            panel.save(out_path)

            share = np.bincount(pred.ravel(), minlength=NUM_CLASSES) / pred.size
            summary = ", ".join(f"{CLASS_NAMES[i]} {share[i]:.1%}" for i in range(NUM_CLASSES) if share[i] > 0)
            print(f"{out_path.name}: {summary}")
            if args.show:
                panel.show()

    print("Saved panels to", args.out)


if __name__ == "__main__":
    main()