from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from visualize import load_model, predict, ROOT, NUM_CLASSES, CLASS_NAMES, EXTS

VAL_SCENES = ["ray1", "seahorse1", "fish11", "fish12"]


def scoring_size(w, h, cap=1024):
    s = min(1.0, cap / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def update(conf, gt, pred):
    enc = NUM_CLASSES * gt.reshape(-1).astype(np.int64) + pred.reshape(-1)
    conf += np.bincount(enc, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(conf):
    tp = np.diag(conf).astype(np.float64)
    gt = conf.sum(1).astype(np.float64)
    pr = conf.sum(0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (gt + pr - tp)
        prec = tp / pr
        rec = tp / gt
        f1 = 2 * tp / (gt + pr)
    present = gt > 0  # classes that exist in the ground truth
    fg_tp = conf[1:, 1:].sum()
    fg_den = conf[1:, :].sum() + conf[:, 1:].sum()
    return dict(
        iou=iou, prec=prec, rec=rec, f1=f1, gt=gt, pr=pr,
        miou=float(np.nanmean(iou[present])),
        macro_f1=float(np.nanmean(f1[present])),
        pixel_acc=float(tp.sum() / conf.sum()),
        fg_f1=float(2 * fg_tp / fg_den) if fg_den > 0 else float("nan"),
    )

def qseg(conf, floor=0.05):
    """Provisional Qseg: macro F1 x mIoU x coverage (0 if a GT-present class has IoU < floor)."""
    m = metrics(conf)
    present = m["gt"] > 0
    coverage = 1.0 if np.all(m["iou"][present] >= floor) else 0.0
    return m["macro_f1"] * m["miou"] * coverage, m["macro_f1"] * m["miou"], coverage


def psnr_norm(psnr_db):
    return float(np.clip((psnr_db - 15.0) / 20.0, 0.0, 1.0))


def composite(q_seg, psnr_db, ssim, sharpness_ratio, nr_quality):
    restoration = 0.45 * psnr_norm(psnr_db) + 0.45 * ssim + 0.10 * sharpness_ratio
    return 0.6 * q_seg + 0.3 * restoration + 0.1 * nr_quality

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder with degraded/ and gt_masks/")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "weighted_best.pt")
    ap.add_argument("--scenes", nargs="*", default=VAL_SCENES)
    ap.add_argument("--max-per-scene", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model(args.ckpt, device)

    total = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    per_scene = {}
    img_q = []

    for scene in args.scenes:
        gt_path = args.root / "gt_masks" / f"{scene}.png"
        imgs = sorted(p for p in (args.root / "degraded" / scene).rglob("*") if p.suffix.lower() in EXTS)
        if args.max_per_scene:
            imgs = imgs[: args.max_per_scene]
        conf = np.zeros_like(total)

        for path in tqdm(imgs, desc=scene, unit="img"):
            image = Image.open(path).convert("RGB")
            size = scoring_size(*image.size)
            image = image.resize(size, Image.Resampling.BICUBIC) if size != image.size else image
            gt = np.asarray(Image.open(gt_path).convert("L").resize(size, Image.Resampling.NEAREST))
            pred = predict(model, image, mean, std, device, out_size=size)
            img_conf = np.zeros_like(total)
            update(img_conf, gt, pred)
            conf += img_conf
            img_q.append(qseg(img_conf)[0])

        per_scene[scene] = metrics(conf)
        total += conf

    m = metrics(total)
    print("\nPer-class (pooled over all pixels)")
    print(f"{'class':<14}{'IoU':>8}{'F1':>8}{'Prec':>8}{'Recall':>8}{'GT px':>14}{'Pred px':>14}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<14}{m['iou'][i]:>8.4f}{m['f1'][i]:>8.4f}{m['prec'][i]:>8.4f}{m['rec'][i]:>8.4f}"
              f"{int(m['gt'][i]):>14,d}{int(m['pr'][i]):>14,d}")

    print("\nPer-scene")
    for s, r in per_scene.items():
        print(f"{s:<14} mIoU {r['miou']:.4f}  macro-F1 {r['macro_f1']:.4f}  pixel-acc {r['pixel_acc']:.4f}")

    print("\nOverall")
    print(f"mIoU (GT classes):        {m['miou']:.4f}")
    print(f"macro F1 (GT classes):    {m['macro_f1']:.4f}")
    print(f"foreground-vs-bg F1:      {m['fg_f1']:.4f}")
    print(f"pixel accuracy:           {m['pixel_acc']:.4f}")
    print(f"F1 x mIoU (macro F1):     {m['macro_f1'] * m['miou']:.4f}   (provisional, no coverage penalty)")
    q, q_free, cov = qseg(total)
    print(f"\nQseg pooled (penalty applied):   {q:.4f}  (without penalty {q_free:.4f}, coverage {cov:.0f})")
    print(f"Qseg mean over images (penalty): {np.mean(img_q):.4f}")
    for s, r in per_scene.items():
        gt_classes = r["gt"] > 0
        print(f"{s:<14} min IoU over GT classes: {np.nanmin(r['iou'][gt_classes]):.3f}")


if __name__ == "__main__":
    main()