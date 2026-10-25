"""
Joint training for TinyEURS (restoration + segmentation), EURS Track 2.

Self-contained: model + training + ONNX export in ONE file.
Kaggle: paste into ONE cell (replacing the old training cell) and commit with Save & Run All.
    train  : EXPORT = False
             - auto-copies last.pth / best.pth from any attached input (e.g. eurs-v2-ckpt)
             - resumes from last.pth
             - writes /kaggle/working/eurs_weights.zip (best + last) after every epoch
    export : EXPORT = True   (or `!python train.py --export`)
"""
import os, glob, math, time, random, argparse, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from multiprocessing import Pool

# ============================ MODEL ============================
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        dt = x.dtype
        x = x.float()  # norm in fp32: avoids fp16 overflow/NaN under AMP
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return ((x - mu) / torch.sqrt(var + 1e-6) * self.w + self.b).to(dt)


class WhiteBalance(nn.Module):
    """Learnable per-channel gain + bias. Removes the blue-green cast cheaply (6 params)."""
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x):
        return torch.clamp(x * self.gain + self.bias, 0.0, 1.0)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.pw2 = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.ffn1 = nn.Conv2d(c, 2 * c, 1)
        self.ffn2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.gate(self.dw(self.pw1(self.norm1(x))))
        y = self.pw2(y * self.sca(y))
        x = x + y * self.beta
        y = self.ffn2(self.gate(self.ffn1(self.norm2(x))))
        return x + y * self.gamma


def stage(c, n):
    return nn.Sequential(*[NAFBlock(c) for _ in range(n)])


class Up(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in * 2, 1, bias=False)  # -> c_in/2 after shuffle
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, widths, dec_blocks):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(Up(widths[i]))
            self.blocks.append(stage(widths[i - 1], dec_blocks[i - 1]))

    def forward(self, x, skips):
        for up, blk, s in zip(self.ups, self.blocks, reversed(skips)):
            x = blk(up(x) + s)
        return x


class TinyEURS(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128), enc_blocks=(1, 1, 2, 4),
                 dec_blocks=(1, 1, 1), n_classes=5):
        super().__init__()
        self.wb = WhiteBalance()
        self.intro = nn.Conv2d(3, widths[0], 3, padding=1)
        self.enc = nn.ModuleList([stage(w, n) for w, n in zip(widths, enc_blocks)])
        self.down = nn.ModuleList(
            [nn.Conv2d(widths[i], widths[i + 1], 2, stride=2) for i in range(len(widths) - 1)])
        self.rest_dec = Decoder(widths, dec_blocks)
        self.seg_dec = Decoder(widths, dec_blocks)
        self.rest_out = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.seg_out = nn.Conv2d(widths[0], n_classes, 3, padding=1)

    def forward(self, inp):
        wb = self.wb(inp)
        x = self.intro(wb)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        delta = 0.5 * torch.tanh(self.rest_out(self.rest_dec(x, skips)))  # bounded correction
        restored = torch.clamp(wb + delta, 0.0, 1.0)
        seg = self.seg_out(self.seg_dec(x, skips))
        return restored, seg


# ============================ CONFIG ============================
DATA_ROOT = "/kaggle/input/datasets/jiyachandna/eurs-train/train"
OUT_DIR = "/kaggle/working/tiny_ckpt"
ZIP_PATH = "/kaggle/working/eurs_weights"   # -> eurs_weights.zip
H, W = 384, 512            # static model input (4:3, same as 4000x3000). ONNX is exported at this size.
NUM_CLASSES = 5            # background, coral, fish, elasmobranch, other
EPOCHS = 100               # keep at 100: resumed LR schedule assumes it
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 3
SPECTRAL_PENALTY_WEIGHT = 0.001
PENALTY_EVERY = 10
PRINT_EVERY = 50
CLIP_NORM = 1.0
SEG_WEIGHT = 1.0           # segmentation is 60% of the official composite
VAL_SCENE = "auto"         # "auto" = scene with most classes whose classes all also appear in train; or a name e.g. "fish9"
RESET_BEST = True          # selection metric changed -> forget old best score when resuming
NUM_WORKERS = 2
SEED = 0
EXPORT = False             # set True to export best.pth -> ONNX instead of training
# ================================================================

IMG_EXT = (".png", ".jpg", ".jpeg")


def find_data_root():
    """Use DATA_ROOT if it exists, else search /kaggle/input for a train/ folder with degraded/ inside."""
    if os.path.isdir(os.path.join(DATA_ROOT, "degraded")):
        return DATA_ROOT
    hits = glob.glob("/kaggle/input/**/degraded", recursive=True)
    if not hits:
        raise FileNotFoundError("no 'degraded' folder found under /kaggle/input")
    root = os.path.dirname(hits[0])
    print("DATA_ROOT not found, using", root)
    return root


def pull_input_checkpoints():
    """Copy last.pth / best.pth from inputs into OUT_DIR (never overwrites).
    Kaggle auto-extracts uploaded .pth files (they are zips) into folders;
    if so, re-zip the folder back into a valid .pth."""
    import zipfile
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ("last.pth", "best.pth"):
        dst = os.path.join(OUT_DIR, name)
        if os.path.exists(dst):
            continue
        src = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if not src:
            print(f"no {name} in inputs (fresh start for this file)")
            continue
        src = src[0]
        if os.path.isdir(src):
            pk = glob.glob(os.path.join(src, "**", "data.pkl"), recursive=True)
            assert pk, f"{src} is a folder but has no data.pkl inside"
            root = os.path.dirname(pk[0])
            prefix = name.replace(".pth", "")
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
                for dp, _, fs in os.walk(root):
                    for f in sorted(fs):
                        full = os.path.join(dp, f)
                        z.write(full, os.path.join(prefix, os.path.relpath(full, root)))
            print(f"rebuilt {dst} from extracted folder {root}")
        else:
            shutil.copy(src, dst)
            print(f"copied {src} -> {dst}")
        ck = torch.load(dst, map_location="cpu", weights_only=False)
        print(f"  {name} OK, epoch {ck['epoch'] + 1}", flush=True)


def write_zip():
    """Zip best.pth + last.pth (+ onnx if present) into /kaggle/working/eurs_weights.zip. Few MB."""
    stage_dir = "/tmp/eurs_weights"
    shutil.rmtree(stage_dir, ignore_errors=True)
    os.makedirs(stage_dir)
    for f in ("best.pth", "last.pth", "tiny_eurs.onnx"):
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            shutil.copy(p, stage_dir)
    shutil.make_archive(ZIP_PATH, "zip", stage_dir)


def load_rgb(path):
    im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR, reducing_gap=2.0)
    return np.asarray(im, dtype=np.uint8)


def load_mask(path):
    m = Image.open(path)
    if m.mode not in ("L", "P"):
        m = m.convert("L")
    return np.asarray(m.resize((W, H), Image.NEAREST), dtype=np.uint8)


def build_cache(data_root):
    """Decode every image once at HxW and keep it in RAM (4000x3000 PNGs are slow to read)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = os.path.join(OUT_DIR, f"cache_{H}x{W}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"loaded cache {cache}")
        return d["deg"], d["sidx"], d["gt"], d["mask"], [str(s) for s in d["scenes"]]

    scenes = sorted(os.listdir(os.path.join(data_root, "degraded")))
    paths, sidx = [], []
    for i, s in enumerate(scenes):
        ps = sorted(p for p in glob.glob(os.path.join(data_root, "degraded", s, "*"))
                    if p.lower().endswith(IMG_EXT))
        paths += ps
        sidx += [i] * len(ps)
        print(f"  scene {s}: {len(ps)} degraded images")

    t = time.time()
    with Pool(os.cpu_count()) as pool:
        deg = np.stack(pool.map(load_rgb, paths, chunksize=8))
    gt = np.stack([load_rgb(os.path.join(data_root, "gt_images", s + ".png")) for s in scenes])
    mask = np.stack([load_mask(os.path.join(data_root, "gt_masks", s + ".png")) for s in scenes])
    sidx = np.array(sidx, dtype=np.int64)
    print(f"decoded {len(paths)} images in {time.time() - t:.0f}s")

    assert mask.max() < NUM_CLASSES, f"mask has class id {mask.max()} >= {NUM_CLASSES}"
    np.savez(cache, deg=deg, sidx=sidx, gt=gt, mask=mask, scenes=np.array(scenes))
    return deg, sidx, gt, mask, scenes


class EURSSet(torch.utils.data.Dataset):
    def __init__(self, deg, sidx, gt, mask, idxs, train):
        self.deg, self.sidx, self.gt, self.mask = deg, sidx, gt, mask
        self.idxs, self.train = idxs, train

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]
        s = self.sidx[i]
        x = torch.from_numpy(self.deg[i]).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(self.gt[s]).permute(2, 0, 1).float() / 255
        m = torch.from_numpy(self.mask[s]).long()
        if self.train:
            if random.random() < 0.5:
                x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
            if random.random() < 0.5:
                sc = random.uniform(0.6, 1.0)
                h, w = int(H * sc), int(W * sc)
                t, l = random.randint(0, H - h), random.randint(0, W - w)
                x = F.interpolate(x[None, :, t:t + h, l:l + w], (H, W), mode="bilinear", align_corners=False)[0]
                y = F.interpolate(y[None, :, t:t + h, l:l + w], (H, W), mode="bilinear", align_corners=False)[0]
                m = F.interpolate(m[None, None, t:t + h, l:l + w].float(), (H, W), mode="nearest")[0, 0].long()
        return x, y, m


def dice_loss(logits, target, eps=1.0):
    p = logits.softmax(1)
    oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    present = oh.sum((0, 2, 3)) > 0
    return 1 - dice[present].mean()


def spectral_penalty(model):
    """Penalize conv weight matrices whose largest singular value exceeds 1."""
    pen = 0.0
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            w = mod.weight.float().reshape(mod.weight.shape[0], -1)
            sigma = torch.linalg.matrix_norm(w, ord=2)
            pen = pen + F.relu(sigma - 1.0) ** 2
    return pen


def class_weights(mask, train_scenes):
    counts = np.bincount(mask[train_scenes].ravel(), minlength=NUM_CLASSES).astype(np.float64)
    freq = counts / counts.sum()
    w = np.where(freq > 0, 1.0 / np.sqrt(freq + 1e-12), 0.0)
    w = w / w[w > 0].mean()
    w[freq == 0] = 1.0
    print("train class pixel freq:", np.round(freq, 4), "| CE weights:", np.round(w, 2))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnrs, conf = [], torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        rest, seg = model(x)
        mse = ((rest.float() - y) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
        psnrs += (10 * torch.log10(1.0 / mse)).tolist()
        pred = seg.argmax(1)
        conf += torch.bincount(m.flatten() * NUM_CLASSES + pred.flatten(),
                               minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    valid = conf.sum(1) > 0
    iou = inter / union.clamp_min(1)
    f1 = 2 * inter / (conf.sum(0).float() + conf.sum(1).float()).clamp_min(1)
    return float(np.mean(psnrs)), iou[valid].mean().item(), f1[valid].mean().item(), iou.tolist()


def pick_val_scene(mask, scenes):
    """Hold out the scene with the most classes, but only if every one of its classes
    also appears in some other scene (so the model is never validated on a class it never saw)."""
    if VAL_SCENE != "auto":
        return scenes.index(VAL_SCENE)
    present = [set(np.unique(m).tolist()) for m in mask]
    best_i, best_n = None, -1
    for i, cls in enumerate(present):
        others = set().union(*(present[j] for j in range(len(scenes)) if j != i))
        if cls <= others and len(cls) > best_n:
            best_i, best_n = i, len(cls)
    for i, cls in enumerate(present):
        print(f"  {scenes[i]:>12}: classes {sorted(cls)}")
    if best_i is None:
        best_i = len(scenes) - 1
    print(f"VAL scene -> {scenes[best_i]} (classes {sorted(present[best_i])})", flush=True)
    return best_i


def official_proxy(psnr, miou, f1):
    """Approximation of the official composite with what we can compute:
    0.6 * Qseg(F1 * mIoU) + 0.3 * restoration(PSNR only). SSIM/sharpness/NR metrics omitted.
    Full metrics: see evaluate.py."""
    return 0.6 * (f1 * miou) + 0.3 * min(psnr / 40, 1.0)


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    assert torch.cuda.is_available(), "No GPU! Set Accelerator to GPU T4 x2 before committing."
    device = "cuda"

    pull_input_checkpoints()
    deg, sidx, gt, mask, scenes = build_cache(find_data_root())
    n_sc = len(scenes)
    v = pick_val_scene(mask, scenes)
    train_sc, val_sc = [i for i in range(n_sc) if i != v], [v]
    tr_idx = np.where(np.isin(sidx, train_sc))[0]
    va_idx = np.where(np.isin(sidx, val_sc))[0]
    assert not set(tr_idx) & set(va_idx)
    print(f"train scenes {[scenes[i] for i in train_sc]} ({len(tr_idx)} imgs) | "
          f"val scenes {[scenes[i] for i in val_sc]} ({len(va_idx)} imgs)")

    tr = torch.utils.data.DataLoader(EURSSet(deg, sidx, gt, mask, tr_idx, True), BATCH_SIZE, shuffle=True,
                                     num_workers=NUM_WORKERS, drop_last=True, pin_memory=True,
                                     persistent_workers=NUM_WORKERS > 0)
    va = torch.utils.data.DataLoader(EURSSet(deg, sidx, gt, mask, va_idx, False), BATCH_SIZE,
                                     num_workers=NUM_WORKERS, pin_memory=True)

    model = TinyEURS(n_classes=NUM_CLASSES).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    cw = class_weights(mask, train_sc).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total, warm = EPOCHS * len(tr), WARMUP_EPOCHS * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")

    start, best = 0, -1e9
    last_path, best_path = os.path.join(OUT_DIR, "last.pth"), os.path.join(OUT_DIR, "best.pth")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, (-1e9 if RESET_BEST else ck["best"])
        print(f"resumed from epoch {start}, best score {best:.4f}", flush=True)
    else:
        print("no last.pth -> training from scratch", flush=True)

    base = [10 * math.log10(1 / max(((deg[i] / 255.0 - gt[sidx[i]] / 255.0) ** 2).mean(), 1e-10)) for i in va_idx]
    print(f"BASELINE val PSNR (no model): {np.mean(base):.2f} dB")

    for epoch in range(start, EPOCHS):
        model.train()
        t0, agg, skipped = time.time(), np.zeros(5), 0
        for step, (x, y, m) in enumerate(tr):
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                rest, seg = model(x)
            l1 = F.l1_loss(rest.float(), y)
            ce = F.cross_entropy(seg.float(), m, weight=cw)
            dl = dice_loss(seg.float(), m)
            pen = spectral_penalty(model) if step % PENALTY_EVERY == 0 else torch.zeros((), device=device)
            loss = l1 + SEG_WEIGHT * (ce + dl) + SPECTRAL_PENALTY_WEIGHT * pen

            if epoch == start and step == 0:
                print(f"[step0] L1={l1.item():.4f} CE={ce.item():.4f} Dice={dl.item():.4f} "
                      f"pen={pen.item():.4f} (weighted {SPECTRAL_PENALTY_WEIGHT * pen.item():.5f})", flush=True)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if not torch.isfinite(gn):
                skipped += 1
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg += np.array([l1.item(), ce.item(), dl.item(), pen.item() * PENALTY_EVERY, float(gn) if torch.isfinite(gn) else 0])
            if (step + 1) % PRINT_EVERY == 0:
                el = time.time() - t0
                print(f"   step {step + 1}/{len(tr)} | {el / (step + 1):.2f} s/step | epoch ETA {el / (step + 1) * (len(tr) - step - 1) / 60:.1f} min | L1 {l1.item():.4f}", flush=True)

        agg /= len(tr)
        psnr, miou, f1, ious = evaluate(model, va, device)
        score = official_proxy(psnr, miou, f1)
        print(f"📉 Epoch {epoch + 1}/{EPOCHS} | L1 {agg[0]:.4f} CE {agg[1]:.3f} Dice {agg[2]:.3f} pen {agg[3]:.3f} "
              f"gradnorm {agg[4]:.2f} | lr {sched.get_last_lr()[0]:.2e} scale {scaler.get_scale():.0f} "
              f"skipped {skipped} | {time.time() - t0:.0f}s", flush=True)
        print(f"📊 Epoch {epoch + 1} | val PSNR {psnr:.2f} dB | mIoU {miou:.4f} | F1 {f1:.4f} | proxy {score:.4f} | IoU/class {[round(v, 3) for v in ious]}", flush=True)

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "psnr": psnr, "miou": miou}, best_path)
            print(f"   ✅ new best (PSNR {psnr:.2f}, mIoU {miou:.4f}) -> best.pth")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, last_path)
        write_zip()   # eurs_weights.zip always has the latest best + last

    print("training done -> /kaggle/working/eurs_weights.zip", flush=True)


def export():
    pull_input_checkpoints()
    ck = torch.load(os.path.join(OUT_DIR, "best.pth"), map_location="cpu", weights_only=False)
    model = TinyEURS(n_classes=NUM_CLASSES).eval()
    model.load_state_dict(ck["model"])
    print(f"best.pth: epoch {ck['epoch'] + 1}, PSNR {ck['psnr']:.2f}, mIoU {ck['miou']:.4f}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    out = os.path.join(OUT_DIR, "tiny_eurs.onnx")
    x = torch.rand(1, 3, H, W)
    kw = dict(input_names=["input"], output_names=["restored_image", "seg_logits"], opset_version=17)
    try:
        torch.onnx.export(model, x, out, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, x, out, **kw)
    print("exported", out)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
        for o, a in zip(sess.get_outputs(), outs):
            print(f"  {o.name}: {a.shape}, range [{a.min():.3f}, {a.max():.3f}]")
    except ImportError:
        print("  (pip install onnxruntime to sanity-check the ONNX file)")
    write_zip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    args, _ = ap.parse_known_args()
    export() if (args.export or EXPORT) else train()
