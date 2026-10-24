import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append('/kaggle/working/Restormer')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.restormer_arch import Restormer
from config import *

TILE = None      # e.g. 512 if full-res OOMs (must be a multiple of 8)
OVERLAP = 64


def pad8(x):
    h, w = x.shape[-2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


def run(model, x, tile, overlap):
    h, w = x.shape[-2:]
    if tile is None or max(h, w) <= tile:
        return model(x)
    stride = tile - overlap
    def starts(n):
        s = list(range(0, max(n - tile, 0) + 1, stride))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s
    out, cnt = torch.zeros_like(x), torch.zeros_like(x)
    for y in starts(h):
        for xx in starts(w):
            out[..., y:y+tile, xx:xx+tile] += model(x[..., y:y+tile, xx:xx+tile])
            cnt[..., y:y+tile, xx:xx+tile] += 1
    return out / cnt


def inference(input_dir, output_dir, ckpt_name='best_restormer.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Restormer(
        inp_channels=3, out_channels=3, dim=RESTORMER_DIM,
        num_blocks=RESTORMER_NUM_BLOCKS, num_refinement_blocks=RESTORMER_NUM_REFINEMENT_BLOCKS,
        heads=RESTORMER_HEADS, ffn_expansion_factor=2.66, bias=False,
        LayerNorm_type='WithBias', dual_pixel_task=False).to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get('model', ck.get('model_state_dict'))
    assert sd is not None, f"no model weights in checkpoint, keys={list(ck)}"
    model.load_state_dict(sd)
    model.eval()
    print(f"✅ Loaded epoch {ck.get('epoch')} (val PSNR {ck.get('psnr', float('nan')):.2f})")

    os.makedirs(output_dir, exist_ok=True)
    scenes = sorted(s for s in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, s)))
    total = 0
    for scene in scenes:
        in_dir, out_dir = os.path.join(input_dir, scene), os.path.join(output_dir, scene)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        print(f"  {scene}: {len(files)} images")
        with torch.no_grad():
            for f in files:
                with Image.open(os.path.join(in_dir, f)) as im:
                    x = TF.to_tensor(im.convert('RGB')).unsqueeze(0).to(device)
                x, h, w = pad8(x)
                pred = run(model, x, TILE, OVERLAP)[..., :h, :w]          # fp32, crop back
                arr = pred.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = Image.fromarray((arr * 255).round().astype(np.uint8))
                assert out.size == (w, h)
                if f.lower().endswith('.png'):
                    out.save(os.path.join(out_dir, f))
                else:
                    out.save(os.path.join(out_dir, f), quality=100, subsampling=0)
                total += 1
    print(f"✅ Done: {total} images -> {output_dir}")


if __name__ == '__main__':
    inference("/kaggle/input/<dataset-slug>/eurs_train/train/degraded",
              "/kaggle/working/restored")
