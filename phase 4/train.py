import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

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
    
    # 1. Datasets & Loaders
    train_ds = EURSRestorationDataset(root_dir=DATA_ROOT, mode='train', patch_size=IMG_SIZE)
    val_ds = EURSRestorationDataset(root_dir=DATA_ROOT, mode='val', patch_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 2. Model (FULL RESTORMER)
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
    scaler = GradScaler() # Crucial for fitting Full Restormer in 16GB
    
    # 4. Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    best_psnr = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 5. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for deg, gt, _ in train_loader:
            deg, gt = deg.to(device), gt.to(device)
            optimizer.zero_grad()
            
            with autocast(): # Mixed Precision
                pred = model(deg)
                loss = criterion(pred, gt)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # 6. Validation Loop
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for deg, gt, _ in val_loader:
                deg, gt = deg.to(device), gt.to(device)
                with autocast():
                    pred = model(deg)
                
                val_psnr += psnr_metric(pred, gt).item()
                val_ssim += ssim_metric(pred, gt).item()
                
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")
        
        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(CHECKPOINT_DIR, 'best_restormer.pth'))
            print(f"  --> 💾 Saved best model (PSNR: {best_psnr:.2f})")

if __name__ == '__main__':
    main()