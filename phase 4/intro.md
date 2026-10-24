# EURS Phase 1 (Restormer) on Kaggle: Setup & Error Handling Guide

This guide covers the `phase4/` code: `config.py`, `dataset.py`, `train.py`, `inference.py`.
**Note:** this code has been reviewed but not yet run end-to-end. Follow the order below so errors show up in minutes, not hours.

---

## 0. Golden rules

1. **Never edit code first.** Read the error message, find it in the tables below, and change only the file/line listed.
2. **`/kaggle/input/` is read-only.** Anything you write (checkpoints, outputs) must go under `/kaggle/working/`.
3. **Always run the preflight check (Section 2) and a smoke test (Section 3) before a full training run.**
4. When stuck, send back: the **full traceback**, the `[step0]` line, and the last few `📉`/`📊` epoch lines.

---

## 1. Kaggle notebook setup

Turn on in the notebook sidebar: **Accelerator → GPU T4** (one GPU is enough) and **Internet → On** (needed for `git clone` and `pip`).

```bash
# Cell 1: dependencies
!pip install -q einops torchmetrics

# Cell 2: Restormer source
%cd /kaggle/working
!git clone https://github.com/swz30/Restormer.git

# Cell 3: your code (upload phase4/ as a Kaggle dataset, then copy it)
!cp -r /kaggle/input/<your-code-dataset>/phase4 /kaggle/working/phase4
%cd /kaggle/working/phase4
```

Expected layout:

```
/kaggle/working/
├── Restormer/basicsr/models/archs/restormer_arch.py
└── phase4/
    ├── config.py
    ├── dataset.py
    ├── train.py
    └── inference.py
```

Run scripts with `!python train.py` from inside `/kaggle/working/phase4`.

---

## 2. Preflight path check (run before anything else)

```python
import os, sys
sys.path.insert(0, '/kaggle/working/phase4')
from config import *

def check(path, kind='dir'):
    ok = os.path.isdir(path) if kind == 'dir' else os.path.isfile(path)
    print(('✅' if ok else '❌'), path)
    return ok

print("Datasets mounted:", os.listdir('/kaggle/input'))
check(DATA_ROOT); check(DEGRADED_DIR); check(GT_DIR)
check('/kaggle/working/Restormer/basicsr/models/archs/restormer_arch.py', 'file')

scenes = sorted(d for d in os.listdir(DEGRADED_DIR) if os.path.isdir(os.path.join(DEGRADED_DIR, d)))
print("Scenes:", scenes)
for s in scenes:
    n = len([f for f in os.listdir(os.path.join(DEGRADED_DIR, s)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    check(os.path.join(GT_DIR, f"{s}.png"), 'file')
    print(f"   {s}: {n} degraded images")

print("CHECKPOINT_DIR:", CHECKPOINT_DIR, "(must start with /kaggle/working)")
for name in ['WEIGHT_DECAY', 'WARMUP_EPOCHS', 'GRAD_CLIP', 'SEED', 'VAL_STRIDE',
             'USE_SPECTRAL_PENALTY', 'SPECTRAL_PENALTY_WEIGHT']:
    print(('✅' if name in globals() else '❌ MISSING in config.py:'), name)
```

**Every line must show ✅ before training.** If one doesn't, fix it with the table below.

### Path errors

| What you see | Cause | Where to fix |
|---|---|---|
| `❌ /kaggle/input/YOUR_DATASET_NAME/...` | Placeholder never replaced | `config.py` → `DATA_ROOT`. Use the name printed by `os.listdir('/kaggle/input')`. |
| `DATA_ROOT` ✅ but `DEGRADED_DIR` ❌ | Kaggle unzipped with an extra or missing folder level | `config.py` → `DATA_ROOT`. Run `!find /kaggle/input -maxdepth 4 -type d` and point `DATA_ROOT` at the folder that directly contains `degraded/` and `gt_images/`. |
| A scene's GT `.png` ❌ | GT name doesn't match scene folder name, or GT is `.jpg` | Rename nothing in `/kaggle/input` (read-only). Edit `dataset.py` → the line `gt_path = os.path.join(self.gt_dir, f"{scene}.png")` to match the real naming. |
| Scene shows `0 degraded images` | Images use an extension not in the list (e.g. `.PNG.tif`) | `dataset.py` → `IMG_EXTS` (and the same tuple in `inference.py`). |
| `CHECKPOINT_DIR` doesn't start with `/kaggle/working` | Relative path from the old Phase 1 config | `config.py` → `CHECKPOINT_DIR = "/kaggle/working/phase4/checkpoints"` |
| `❌ MISSING in config.py: ...` | Teammate is using the old Phase 1 `config.py` | `config.py` → add the missing constants (Section 6). |
| `❌ .../restormer_arch.py` | Restormer not cloned, or cloned elsewhere | Re-run Cell 2, or fix the `sys.path.append(...)` line at the top of `train.py` **and** `inference.py`. |

---

## 3. Smoke test (about 5 minutes)

Temporarily set in `config.py`:

```python
NUM_EPOCHS = 2
IMAGES_PER_EPOCH = 40
WARMUP_EPOCHS = 1
CHECKPOINT_INTERVAL = 1
```

Run `!python train.py`. It passes if you see, in order:

1. `[*] train: scenes [...]` and `[*] val: scenes [...]` with **no scene in both lists**
2. `[*] Val baseline PSNR (degraded vs GT): XX.XX dB`
3. `[step0] deg[0.00,1.00] gt[0.00,1.00] L1=... pen=...`
4. Two `📉 Epoch` lines and two `📊 Epoch` lines
5. Files in `CHECKPOINT_DIR`: `best_restormer.pth`, `last.pth`, `checkpoint_epoch_1.pth`

Then run `!python train.py` **again without deleting anything**. It must print `🔁 resumed from epoch 2`. That proves resume works before you rely on it.

**Delete `CHECKPOINT_DIR` and restore the real config values before the full run**, otherwise training resumes from the smoke-test checkpoint.

```bash
!rm -rf /kaggle/working/phase4/checkpoints
```

---

## 4. Error lookup by file

### 4.1 Import errors (top of `train.py` / `inference.py`)

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'basicsr'` | Restormer path wrong | `train.py` and `inference.py` → `sys.path.append('/kaggle/working/Restormer')` must point to the cloned repo. |
| `ModuleNotFoundError: einops` / `torchmetrics` | Not installed | Run `!pip install -q einops torchmetrics` |
| `ModuleNotFoundError` for anything else (`lmdb`, `cv2`, `skimage`, …) while importing `basicsr` | `basicsr` package `__init__` pulls in extra libraries | **Fallback:** `!cp /kaggle/working/Restormer/basicsr/models/archs/restormer_arch.py /kaggle/working/phase4/`, then in `train.py` and `inference.py` replace `from basicsr.models.archs.restormer_arch import Restormer` with `from restormer_arch import Restormer`. This file only needs torch and einops. |
| `ImportError` with a pip-installed `basicsr` in the traceback path | Kaggle's preinstalled `basicsr` shadows the Restormer one | Change `sys.path.append(...)` to `sys.path.insert(0, ...)` in both scripts, or use the fallback above. |
| `ModuleNotFoundError: config` / `dataset` | Script not run from `phase4/` | `%cd /kaggle/working/phase4` before `!python train.py`. |
| `NameError: name 'WEIGHT_DECAY' is not defined` (or `GRAD_CLIP`, `SEED`, `VAL_STRIDE`, `WARMUP_EPOCHS`, `SPECTRAL_PENALTY_WEIGHT`) | Old Phase 1 `config.py` | `config.py` → add constants from Section 6. |
| `AttributeError: module 'torch.amp' has no attribute 'GradScaler'` | Old PyTorch (< 2.3) | `train.py` → change to `from torch.cuda.amp import GradScaler` and `scaler = GradScaler(enabled=use_amp)`. |

### 4.2 Dataset errors (`dataset.py`)

These are deliberate checks. The message tells you exactly what's wrong.

| Error message | Meaning | Fix |
|---|---|---|
| `FileNotFoundError: Missing GT for scene 'X'` | No `gt_images/X.png` | Check GT naming; edit the `gt_path` line in `dataset.py` if GTs use a different pattern. |
| `ValueError: No samples for mode=...` | Scene folders empty, or extensions not matched | Check `IMG_EXTS` in `dataset.py`; check `DATA_ROOT`. |
| `AssertionError: not enough scenes` | Fewer than 2 scene folders found | `DATA_ROOT` points to the wrong level. |
| `AssertionError: size mismatch scene_img.png: (W1,H1) vs (W2,H2)` | A degraded image and its GT differ in size. Training on it would silently misalign pixels. | **Do not delete the assert.** Report which scenes are affected. If only some images, skip them in the sample-building loop in `dataset.py`; if the whole dataset, they must be resized to match (ask before doing this). |
| `AssertionError: ... smaller than patch 256` | Image smaller than `IMG_SIZE` | Lower `IMG_SIZE` in `config.py` to a multiple of 8 that fits (e.g. 192), or report it. |
| `AssertionError: LEAKAGE` (in `train.py`) | Same image in train and val | Should be impossible with the scene split. Report it; do not bypass. |
| `DataLoader worker (pid X) is killed` / worker crash with unclear error | Real error hidden inside a worker | `config.py` → `NUM_WORKERS = 0` and rerun. The real traceback appears. Restore to 2 afterwards. |

### 4.3 Training errors (`train.py`)

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.OutOfMemoryError` | Batch or crop too big | `config.py`: first `BATCH_SIZE = 2`; if still OOM, `IMG_SIZE = 192`. Keep `IMG_SIZE` a multiple of 8. |
| OOM only during validation | Same, at val time | `config.py` → raise `VAL_STRIDE` (doesn't fix memory) or lower `IMG_SIZE`. Val runs at batch 1 on `IMG_SIZE` crops, so this is rare. |
| `[step0]` weighted penalty is more than ~30% of `L1` | Spectral penalty dominates the loss | `config.py` → lower `SPECTRAL_PENALTY_WEIGHT` (0.01 → 0.003 → 0.001). **Don't turn the penalty off.** |
| `skipped` > 0 in epoch lines | Non-finite loss batches were dropped | A few in epoch 1 is tolerable. Growing every epoch: report the logs. |
| `scale` keeps halving each epoch (e.g. 65536 → 1024 → 16) | fp16 overflow keeps happening | Report the logs. Temporary test: `config.py` → `USE_AMP = False` to see whether it's AMP-related. |
| `gradnorm(pre-clip)` trending upward epoch after epoch | Instability building | Report the logs before it turns into NaN. |
| `L1` is `nan` | Explosion despite safeguards | Report the full log. Don't just lower the LR, since we need to know which term blew up. |
| Val PSNR stays at or below the baseline PSNR | Model not learning | Check `[step0]` ranges are `[0,1]`; check LR reaches 1e-4 by epoch `WARMUP_EPOCHS`. Report logs. |
| Train L1 keeps dropping, val PSNR flat or falling | Overfitting (expected risk: only 5 training GT images) | Report logs. Use `best_restormer.pth`, not the last epoch. |
| `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()` | AMP skipped a step | Harmless. Ignore. |
| Training stopped (12h Kaggle limit, disconnect) | Session ended | Rerun `!python train.py`. It resumes from `last.pth` automatically, as long as `/kaggle/working` was saved (use **Save Version → Save & Run All**, or download checkpoints). |
| `OSError: [Errno 28] No space left on device` | `/kaggle/working` (~20 GB) is full | Delete old `checkpoint_epoch_*.pth`, or raise `CHECKPOINT_INTERVAL`. Each periodic file is ~100 MB; `last.pth` is ~300 MB. |
| `OSError: Read-only file system` | Writing to `/kaggle/input` | `config.py` → `CHECKPOINT_DIR` under `/kaggle/working`. |

### 4.4 Inference errors (`inference.py`)

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Checkpoint not found` | Wrong `CHECKPOINT_DIR`, or using a checkpoint from a different session | Point `CHECKPOINT_DIR` in `config.py` to where `best_restormer.pth` is, e.g. an uploaded Kaggle dataset path. |
| `AssertionError: no model weights in checkpoint, keys=[...]` | Checkpoint saved with an unknown key | `inference.py` → the `sd = ck.get(...)` line; add the key name printed in the error. |
| `RuntimeError: Error(s) in loading state_dict ... size mismatch` | Architecture in `config.py` differs from the one used in training | Make `RESTORMER_DIM`, `RESTORMER_NUM_BLOCKS`, `RESTORMER_HEADS`, `RESTORMER_NUM_REFINEMENT_BLOCKS` match training exactly. |
| `Unexpected key(s) ... "module.xxx"` | Model was trained wrapped in DataParallel | In `inference.py` after `sd = ...`: `sd = {k.replace('module.', '', 1): v for k, v in sd.items()}` |
| `RuntimeError: Sizes of tensors must match` | Image not padded to a multiple of 8 | Shouldn't happen with `pad8()`. If it does, report the image size. |
| `torch.cuda.OutOfMemoryError` on large images | Full resolution too big for T4 | `inference.py` → `TILE = 512` (try 256 if still OOM). Must be a multiple of 8. |
| `_pickle.UnpicklingError` / `weights_only` error | PyTorch ≥ 2.6 safer default for `torch.load` | `inference.py` and `train.py` → `torch.load(path, map_location=device, weights_only=False)` (only for your own checkpoints). |
| Output images look dark/washed out vs expected | Usually a training issue, not inference | First run inference on the **val scene** and compare PSNR to the training val PSNR. Report both numbers. |

**Before trusting inference output:** run it on the val scene, check one output image by eye, and confirm its size equals its input.

---

## 5. What NOT to change

- Don't remove the spectral penalty, warmup, gradient clipping, or `clamp(0, 1)`. That's the agreed hybrid approach.
- Don't remove the `assert` lines in `dataset.py` to "make the error go away". They catch silent data bugs.
- Don't change the scene split to a random image split. That leaks data.
- Don't change the `scaler.unscale_` → `clip_grad_norm_` → `scaler.step` order.

---

## 6. Required `config.py` (Phase 4)

```python
import os

DATA_ROOT = "/kaggle/input/<actual-dataset-slug>/eurs_train/train"
DEGRADED_DIR = os.path.join(DATA_ROOT, 'degraded')
GT_DIR = os.path.join(DATA_ROOT, 'gt_images')
CHECKPOINT_DIR = "/kaggle/working/phase4/checkpoints"

IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
NUM_WORKERS = 2
SEED = 42
VAL_STRIDE = 4

IMAGES_PER_EPOCH = 300
MEMORY_WINDOW = 100

RESTORMER_DIM = 48
RESTORMER_NUM_BLOCKS = [4, 6, 6, 8]
RESTORMER_NUM_REFINEMENT_BLOCKS = 4
RESTORMER_HEADS = [1, 2, 4, 8]

USE_SPECTRAL_PENALTY = True
SPECTRAL_PENALTY_WEIGHT = 0.01
USE_AMP = True
CHECKPOINT_INTERVAL = 10
```

---

## 7. What to send back when something fails

1. The **full traceback** (not just the last line)
2. Output of the preflight check (Section 2)
3. The `[step0]` line and all `📉` / `📊` epoch lines so far
4. Which file you changed, if any, and what you changed



         
