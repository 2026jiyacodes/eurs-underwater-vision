import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add Restormer submodule to path
sys.path.append(os.path.abspath('Restormer'))
from basicsr.models.archs.restormer_arch import Restormer

from config import *
from dataset import EURSRestorationDataset

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # 1. Datasets & Loaders (Scene-based split, no leakage)
    train_ds = EURSRestorationDataset(root_dir=DATA_ROOT, mode='train', patch_size=IMG_SIZE)
    val_ds = EURSRestorationDataset(root_dir=DATA_ROOT, mode='val', patch_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"✅ Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # 2. Model (Full Restormer dim=48 for 16GB VRAM)
    model = Restormer(
        inp_channels=3, out_channels=3, dim=RESTORMER_DIM,
        num_blocks=RESTORMER_NUM_BLOCKS, 
        num_refinement_blocks=RESTORMER_NUM_REFINEMENT_BLOCKS,
        heads=RESTORMER_HEADS,
        ffn_expansion_factor=2.66, bias=False, 
        LayerNorm_type='WithBias', dual_pixel_task=False
    ).to(device)
    
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Loss, Optimizer, Scheduler, AMP Scaler
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()  # CRITICAL: Mixed Precision for 16GB VRAM
    
    # 4. Metrics (PSNR/SSIM - what the challenge actually evaluates)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    best_psnr = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 5. Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for deg, gt, _ in pbar:
            deg, gt = deg.to(device), gt.to(device)
            optimizer.zero_grad()
            
            # CRITICAL: Mixed Precision Forward Pass
            with autocast():
                pred = model(deg)
                loss = criterion(pred, gt)
            
            # CRITICAL: Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # 6. Validation Loop with PSNR/SSIM
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for deg, gt, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                deg, gt = deg.to(device), gt.to(device)
                
                with autocast():
                    pred = model(deg)
                
                val_loss += criterion(pred, gt).item()
                val_psnr += psnr_metric(pred, gt).item()
                val_ssim += ssim_metric(pred, gt).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        print(f"📊 Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | PSNR: {avg_val_psnr:.2f} | SSIM: {avg_val_ssim:.4f}")
        
        # 7. Save Best Checkpoint (based on PSNR, not loss)
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_restormer.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': avg_val_ssim,
            }, checkpoint_path)
            print(f"💾 Saved new best model (PSNR: {best_psnr:.2f})")

    print(f"🎉 Training complete! Best PSNR: {best_psnr:.2f}")

if __name__ == '__main__':
    main()
