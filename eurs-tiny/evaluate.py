"""
Full local evaluation of TinyEURS checkpoints against the EURS Track 2 composite.

    composite = 0.6 * Qseg + 0.3 * (0.45 * PSNRnorm + 0.45 * SSIM + 0.10 * sharpness) + 0.1 * NRquality
    Qseg      = F1 * mIoU * coverage

Computes everything except NRquality (UIQM/UCIQE/CCF): the official normalization
ranges are not public here, so the score below is the composite WITHOUT that 10% term.
PSNRnorm = PSNR / 40, capped at 1 (assumed; check the official Evaluation Metrics page).

Usage (Kaggle: save train.py + evaluate.py into /kaggle/working first):
    python evaluate.py                                  # best.pth and last.pth in tiny_ckpt/
    python evaluate.py --ckpt a.pth b.pth               # any checkpoints
    python evaluate.py --val-scene fish9                # force a val scene
"""
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F

import train as T  # reuses model, data cache and val-scene picking (train() does not run on import)


# ---------------------------- restoration metrics ----------------------------
def _gauss_win(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim_batch(a, b, win):
    """Standard SSIM (11x11 Gaussian, sigma 1.5, data range 1), averaged over RGB. Returns (B,)."""
    C1, C2, pad = 0.01 ** 2, 0.03 ** 2, win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    s_aa = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=pad, groups=3) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return m.mean((1, 2, 3))


_LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
_GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def sharpness_ratio_batch(rest, gt):
    """Laplacian-variance sharpness of restored / GT (grayscale), capped at 1 so
    over-sharpening and noise are not rewarded. 1.0 = as sharp as GT. Returns (B,).
    The official definition may differ; this is the common anti-blur formulation."""
    lap, w = _LAP.to(rest.device), _GRAY.to(rest.device)
    vr = F.conv2d((rest * w).sum(1, keepdim=True), lap).var((1, 2, 3))
    vg = F.conv2d((gt * w).sum(1, keepdim=True), lap).var((1, 2, 3))
    return (vr / vg.clamp_min(1e-8)).clamp(max=1.0)


# ---------------------------------- eval ----------------------------------
@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    win = _gauss_win(device=device)
    psnrs, ssims, sharps = [], [], []
    conf = torch.zeros(T.NUM_CLASSES, T.NUM_CLASSES, dtype=torch.long, device=device)
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        rest, seg = model(x)
        rest = rest.float()
        mse = ((rest - y) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
        psnrs += (10 * torch.log10(1.0 / mse)).tolist()
        ssims += ssim_batch(rest, y, win).tolist()
        sharps += sharpness_ratio_batch(rest, y).tolist()
        pred = seg.argmax(1)
        conf += torch.bincount(m.flatten() * T.NUM_CLASSES + pred.flatten(),
                               minlength=T.NUM_CLASSES ** 2).view(T.NUM_CLASSES, T.NUM_CLASSES)

    inter = conf.diag().float()
    gt_cnt, pred_cnt = conf.sum(1).float(), conf.sum(0).float()
    valid = gt_cnt > 0                                    # classes present in val GT
    iou = inter / (gt_cnt + pred_cnt - inter).clamp_min(1)
    f1 = 2 * inter / (gt_cnt + pred_cnt).clamp_min(1)

    r = dict(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), sharp=float(np.mean(sharps)),
             miou=iou[valid].mean().item(), f1=f1[valid].mean().item(),
             coverage=(pred_cnt[valid] > 0).float().mean().item(),
             ious=[round(v, 3) for v in iou.tolist()], present=valid.tolist())
    r["psnr_norm"] = min(r["psnr"] / 40, 1.0)
    r["qseg"] = r["f1"] * r["miou"] * r["coverage"]
    r["rest"] = 0.45 * r["psnr_norm"] + 0.45 * r["ssim"] + 0.10 * r["sharp"]
    r["composite_no_nr"] = 0.6 * r["qseg"] + 0.3 * r["rest"]
    return r


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = T.TinyEURS(n_classes=T.NUM_CLASSES).to(device)
    model.load_state_dict(ck["model"])
    return model, ck.get("epoch", -1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=[os.path.join(T.OUT_DIR, "best.pth"), os.path.join(T.OUT_DIR, "last.pth")])
    ap.add_argument("--val-scene", default=None, help="override train.VAL_SCENE (default: same as training)")
    args, _ = ap.parse_known_args()
    if args.val_scene:
        T.VAL_SCENE = args.val_scene

    device = "cuda" if torch.cuda.is_available() else "cpu"
    deg, sidx, gt, mask, scenes = T.build_cache(T.find_data_root())
    v = T.pick_val_scene(mask, scenes)
    va_idx = np.where(sidx == v)[0]
    loader = torch.utils.data.DataLoader(T.EURSSet(deg, sidx, gt, mask, va_idx, False), T.BATCH_SIZE,
                                         num_workers=T.NUM_WORKERS)
    print(f"val scene: {scenes[v]} ({len(va_idx)} imgs) | device {device}\n")

    results = []
    for path in args.ckpt:
        if not os.path.exists(path):
            print(f"skip {path} (not found)")
            continue
        model, ep = load_model(path, device)
        r = evaluate_full(model, loader, device)
        results.append((path, ep, r))
        print(f"== {path} (epoch {ep})")
        print(f"   restoration: PSNR {r['psnr']:.2f} dB (norm {r['psnr_norm']:.3f}) | SSIM {r['ssim']:.4f} | "
              f"sharp {r['sharp']:.3f}  ->  {r['rest']:.4f}")
        print(f"   segmentation: mIoU {r['miou']:.4f} | F1 {r['f1']:.4f} | coverage {r['coverage']:.2f}  ->  Qseg {r['qseg']:.4f}")
        print(f"   IoU/class {r['ious']} (present in val: {r['present']})")
        print(f"   COMPOSITE (without 10% NR term): {r['composite_no_nr']:.4f}\n")

    if len(results) > 1:
        best = max(results, key=lambda t: t[2]["composite_no_nr"])
        print(f"→ use {best[0]} (epoch {best[1]}, composite {best[2]['composite_no_nr']:.4f}) for ONNX export")


if __name__ == "__main__":
    main()
