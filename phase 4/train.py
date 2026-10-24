import os, sys, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append('/kaggle/working/Restormer')          # where the submodule is cloned
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.restormer_arch import Restormer

from config import *
from dataset import EURSRestorationDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SpectralPenalty:
    """mean over conv layers of relu(sigma_max - 1)^2; one power iteration per call."""
    def __init__(self, model, init_iters=20):
        self.weights = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.u = [F.normalize(torch.randn(w.shape[0], device=w.device), dim=0) for w in self.weights]
        with torch.no_grad():
            for _ in range(init_iters):
                for w, u in zip(self.weights, self.u):
                    self._power(w.float().reshape(w.shape[0], -1), u)

    @staticmethod
    def _power(W, u):
        v = F.normalize(W.t() @ u, dim=0)
        u.copy_(F.normalize(W @ v, dim=0))
        return v

    def __call__(self):
        terms = []
        for w, u in zip(self.weights, self.u):
            W = w.float().reshape(w.shape[0], -1)
            with torch.no_grad():
                v = self._power(W, u)
            sigma = u @ (W @ v)                 # gradient flows through W only
            terms.append(F.relu(sigma - 1.0) ** 2)
        return torch.stack(terms).mean()


def sample_epoch_indices(n, memory, k):
    mem = list(memory)
    mem_set = set(mem)
    need = max(0, min(k - len(mem), n - len(mem)))
    pool = [i for i in range(n) if i not in mem_set]
    idx = mem + random.sample(pool, need)
    random.shuffle(idx)
    return idx


def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = USE_AMP and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True
    print(f"🚀 device={device} amp={use_amp}")

    # --- data ---
    train_ds = EURSRestorationDataset(DATA_ROOT, 'train', IMG_SIZE)
    val_ds = EURSRestorationDataset(DATA_ROOT, 'val', IMG_SIZE)
    assert not {s['deg'] for s in train_ds.samples} & {s['deg'] for s in val_ds.samples}, "LEAKAGE"
    val_loader = DataLoader(Subset(val_ds, range(0, len(val_ds), VAL_STRIDE)),
                            batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- model ---
    model = Restormer(
        inp_channels=3, out_channels=3, dim=RESTORMER_DIM,
        num_blocks=RESTORMER_NUM_BLOCKS, num_refinement_blocks=RESTORMER_NUM_REFINEMENT_BLOCKS,
        heads=RESTORMER_HEADS, ffn_expansion_factor=2.66, bias=False,
        LayerNorm_type='WithBias', dual_pixel_task=False).to(device)
    print(f"[*] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- optimizer: no decay on norm params / attention temperature ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or 'temperature' in n else decay).append(p)
    optimizer = optim.AdamW([{'params': decay, 'weight_decay': WEIGHT_DECAY},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=LEARNING_RATE)

    # --- per-iteration linear warmup -> cosine ---
    steps_per_epoch = IMAGES_PER_EPOCH // BATCH_SIZE
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * t)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = nn.L1Loss()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # --- resume ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    last_path = os.path.join(CHECKPOINT_DIR, 'last.pth')
    start_epoch, best_psnr, memory_buffer = 1, 0.0, []
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        start_epoch, best_psnr, memory_buffer = ck['epoch'] + 1, ck['best_psnr'], ck['memory']
        print(f"🔁 resumed from epoch {ck['epoch']}")

    penalty = SpectralPenalty(model) if USE_SPECTRAL_PENALTY else None

    # --- val baseline: PSNR of the raw degraded input (model must beat this) ---
    with torch.no_grad():
        base = [psnr_metric(d.to(device), g.to(device)).item() for d, g, _ in val_loader]
    print(f"[*] Val baseline PSNR (degraded vs GT): {sum(base)/len(base):.2f} dB")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        idx = sample_epoch_indices(len(train_ds), memory_buffer, IMAGES_PER_EPOCH)
        assert len(set(idx)) == len(idx)
        memory_buffer = idx[-MEMORY_WINDOW:]
        loader = DataLoader(Subset(train_ds, idx), batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)

        s = dict(l1=0.0, pen=0.0, gn=0.0, n=0, skipped=0)
        for deg, gt, _ in tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            deg, gt = deg.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', enabled=use_amp):
                pred = model(deg)
            l1 = criterion(pred.float(), gt)                       # fp32
            pen = penalty() if penalty else torch.zeros((), device=device)  # fp32, outside autocast
            loss = l1 + SPECTRAL_PENALTY_WEIGHT * pen

            if s['n'] == 0 and epoch == start_epoch:
                print(f"[step0] deg[{deg.min():.2f},{deg.max():.2f}] gt[{gt.min():.2f},{gt.max():.2f}] "
                      f"L1={l1.item():.4f} pen={pen.item():.4f} "
                      f"(weighted {SPECTRAL_PENALTY_WEIGHT*pen.item():.4f})")

            if not torch.isfinite(loss):
                s['skipped'] += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                              # clip true grads
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not use_amp and not torch.isfinite(gn):              # AMP: scaler handles inf itself
                s['skipped'] += 1
                continue
            scaler.step(optimizer); scaler.update(); scheduler.step()

            s['l1'] += l1.item(); s['pen'] += pen.item(); s['gn'] += gn.item(); s['n'] += 1

        n = max(1, s['n'])
        print(f"📉 Epoch {epoch} | L1 {s['l1']/n:.4f} | pen {s['pen']/n:.4f} | "
              f"gradnorm(pre-clip) {s['gn']/n:.3f} | lr {scheduler.get_last_lr()[0]:.2e} | "
              f"scale {scaler.get_scale():.0f} | skipped {s['skipped']}")

        # --- validation (deterministic center crops, clamped, fp32 metrics) ---
        model.eval(); ps = ss = 0.0
        with torch.no_grad():
            for deg, gt, _ in val_loader:
                deg, gt = deg.to(device), gt.to(device)
                with torch.autocast('cuda', enabled=use_amp):
                    pred = model(deg)
                pred = pred.float().clamp(0, 1)
                ps += psnr_metric(pred, gt).item(); ss += ssim_metric(pred, gt).item()
        v_psnr, v_ssim = ps / len(val_loader), ss / len(val_loader)
        print(f"📊 Epoch {epoch} | Val PSNR {v_psnr:.2f} | Val SSIM {v_ssim:.4f}")

        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'psnr': v_psnr, 'ssim': v_ssim},
                       os.path.join(CHECKPOINT_DIR, 'best_restormer.pth'))
            print(f"💾 best saved (PSNR {best_psnr:.2f})")

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'psnr': v_psnr},
                       os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'best_psnr': best_psnr, 'memory': memory_buffer}, last_path)

    print(f"🎉 done. best PSNR {best_psnr:.2f}")


if __name__ == '__main__':
    main()
