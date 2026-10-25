# TinyEURS: joint underwater restoration + segmentation

A tiny multi-task model for **EURS Track 2** (Underwater Restoration and Segmentation).
One shared encoder, two decoders: one restores the degraded image, one segments it into 5 classes.

- **761K parameters**, NAFNet-style blocks (no attention), exports cleanly to ONNX
- Fixed input **384×512** (4:3, same as the 4000×3000 originals)
- Classes: `0 background · 1 coral · 2 fish · 3 elasmobranch · 4 other`

## Files

| File | Purpose |
|---|---|
| `train.py` | Model, data cache, training with resume, ONNX export, weights zip (all-in-one) |
| `evaluate.py` | Full local metrics on checkpoints: PSNR, SSIM, sharpness, mIoU, F1, coverage, composite |
| `model.py` | Model definition alone, plus a parameter/shape sanity check |
| `inspect_data.py` | Prints the dataset layout, a sample image size and mask values |

## Data layout

```
train/
├── degraded/<scene>/*.png   # 200 degraded versions per scene, 20 scenes
├── gt_images/<scene>.png    # one clean image per scene
└── gt_masks/<scene>.png     # one mask per scene, values 0–4
```

Every degraded image of a scene shares that scene's single GT image and mask.
One scene is held out for validation. It is chosen automatically as the scene with the most classes
whose classes all also appear in the training scenes.

## Model

```
input ─► learnable white balance ─► intro conv
       ─► encoder: 4 stages (16→32→64→128 ch), NAFBlocks, strided-conv downsampling
       ├─► restoration decoder ─► bounded residual (0.5·tanh) added to white-balanced input
       └─► segmentation decoder ─► 5-class logits
```

Each NAFBlock: LayerNorm → 1×1 conv → depthwise 3×3 → SimpleGate (a·b) → channel reweighting → 1×1 conv, then a gated feed-forward.

## Training

```
loss = L1(restored, gt) + SEG_WEIGHT · (weighted CE + Dice) + 0.001 · spectral penalty
```

- AdamW, LR 1e-3, 3 warmup epochs, then cosine decay to 1% over 100 epochs, AMP fp16, grad clip 1.0
- Rare classes get larger CE weights (inverse √frequency)
- Augmentation: horizontal flip, random crop + resize
- Images are decoded once into an in-RAM cache (`cache_384x512.npz`, ~2.2 GB)
- Resumes automatically from `tiny_ckpt/last.pth`. On Kaggle it also pulls `last.pth`/`best.pth` from any
  attached input dataset, rebuilding them if Kaggle auto-extracted the `.pth` archives into folders.
- `best.pth` is selected each epoch by a proxy of the official score: `0.6·F1·mIoU + 0.3·PSNR/40`

## Official composite (Track 2)

```
composite = 0.6·Qseg + 0.3·(0.45·PSNRnorm + 0.45·SSIM + 0.10·sharpness) + 0.1·NRquality
Qseg      = F1 · mIoU · coverage_penalty
NRquality = mean of normalized UIQM / UCIQE / CCF
```

`evaluate.py` computes everything except NRquality. PSNRnorm is assumed to be PSNR/40, and sharpness
is a Laplacian-variance ratio vs. GT. Use the organizers' evaluation code for exact leaderboard numbers.

## Usage (Kaggle)

1. Attach the `eurs-train` dataset (and, to resume, a dataset with `last.pth` / `best.pth`).
2. Accelerator: GPU. Paste `train.py` into one cell, then **Save Version → Save & Run All**.
3. Download `eurs_weights.zip` from the version's Output.
4. Export: set `EXPORT = True`, attach the trained output, and commit again to get `tiny_eurs.onnx`.
5. Evaluate: write `train.py` and `evaluate.py` to `/kaggle/working` (e.g. `%%writefile`), then `!python evaluate.py`.

## Requirements

```
pip install -r requirements.txt
```
