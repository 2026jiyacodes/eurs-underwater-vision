"""Print the EURS dataset layout, one sample image size, and one mask's values.
Run on Kaggle with the eurs-train dataset attached."""
import os, numpy as np
from PIL import Image

print("Inputs:", os.listdir("/kaggle/input"))
ROOT = os.path.join("/kaggle/input", os.listdir("/kaggle/input")[0])
print("ROOT =", ROOT, "\n")

# 1. folder layout (4 levels deep, first 3 files per folder)
for dirpath, dirs, files in os.walk(ROOT):
    depth = dirpath[len(ROOT):].count(os.sep)
    if depth > 3:
        continue
    print("  " * depth + os.path.basename(dirpath) + f"/  [{len(dirs)} dirs, {len(files)} files]")
    for f in sorted(files)[:3]:
        print("  " * (depth + 1) + f)

# 2. one image's size
for dirpath, _, files in os.walk(ROOT):
    imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if imgs:
        im = Image.open(os.path.join(dirpath, imgs[0]))
        print("\nSAMPLE IMAGE:", os.path.join(dirpath, imgs[0]), im.size, im.mode)
        break

# 3. inspect one mask
found = False
for dirpath, _, files in os.walk(ROOT):
    if any(k in dirpath.lower() for k in ("mask", "seg", "label", "ann")):
        imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".npy"))]
        if imgs:
            p = os.path.join(dirpath, sorted(imgs)[0])
            m = np.load(p) if p.endswith(".npy") else np.array(Image.open(p))
            print("\nMASK:", p, "| shape", m.shape, "| dtype", m.dtype, "| unique", np.unique(m)[:20])
            found = True
            break
if not found:
    print("\nNO MASK FOLDER FOUND")
