import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
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
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"✅ Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # 2. Model (Full Restormer dim=48)
    model = Restormer(
        inp_channels=3, out_channels=3, dim=RESTORMER_DIM,
        num_blocks=RESTORMER_NUM_BLOCKS, 
        num_refinement_blocks=RESTORMER_NUM_REFINEMENT_BLOCKS,
        heads=RESTORMER_HEADS,
        ffn_expansion_factor=2.66, bias=False, 
        LayerNorm_type='WithBias', dual_pixel_task=False
    ).to(device)
    
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Loss, Optimizer, Scheduler
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # 4. Mixed Precision Setup (for 16GB VRAM)
    scaler = GradScaler() if USE_AMP else None
    
    # 5. Metrics (PSNR/SSIM)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    best_psnr = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 6. Mini-Epoch Memory Buffer
    memory_buffer = []
    
    # 7. Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # --- MINI-EPOCH SAMPLING ---
        all_indices = list(range(len(train_ds)))
        new_samples_needed = min(IMAGES_PER_EPOCH - len(memory_buffer), len(all_indices) - len(memory_buffer))
        
        if new_samples_needed > 0:
            available_indices = [i for i in all_indices if i not in memory_buffer]
            new_random_indices = random.sample(available_indices, new_samples_needed)
            current_epoch_indices = memory_buffer + new_random_indices
        else:
            current_epoch_indices = memory_buffer
        
        random.shuffle(current_epoch_indices)
        
        mini_train_ds = Subset(train_ds, current_epoch_indices)
        mini_train_loader = DataLoader(
            mini_train_ds, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # Keep only the last MEMORY_WINDOW items for next epoch
        memory_buffer = current_epoch_indices[-MEMORY_WINDOW:]
        # ---------------------------
        
        pbar = tqdm(mini_train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for deg, gt, _ in pbar:
            deg, gt = deg.to(device), gt.to(device)
            
            # 🔍 SANITY CHECK: Verify data is valid
            if torch.isnan(deg).any() or torch.isnan(gt).any():
                print("❌ NaN found in input data!")
                continue
            if deg.max() > 1.0 or gt.max() > 1.0:
                print(f"⚠️ Data values > 1.0: deg_max={deg.max():.4f}, gt_max={gt.max():.4f}")
            if deg.min() < 0.0 or gt.min() < 0.0:
                print(f"⚠️ Data values < 0.0: deg_min={deg.min():.4f}, gt_min={gt.min():.4f}")
            
            optimizer.zero_grad()
            
            # 🔥 USE AMP (Mixed Precision) for 16GB VRAM
            if USE_AMP:
                with autocast():
                    pred = model(deg)
                    loss = criterion(pred, gt)
                
                # 🔍 SANITY CHECK: Verify model output
                if torch.isnan(pred).any():
                    print("❌ Model output contains NaN!")
                    continue
                if torch.isinf(pred).any():
                    print("❌ Model output contains Inf!")
                    continue
                
                # 🔍 SANITY CHECK: Verify loss is valid
                loss_val = loss.item()
                if loss_val < 0:
                    print(f"❌ NEGATIVE LOSS DETECTED: {loss_val}")
                    raise ValueError("Loss is negative!")
                if math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"❌ Invalid loss value: {loss_val}")
                    continue
                
                scaler.scale(loss).backward()
                
                # 🔥 CRITICAL: Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # No AMP (FP32)
                pred = model(deg)
                
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print("❌ Model output contains NaN/Inf!")
                    continue
                
                loss = criterion(pred, gt)
                loss_val = loss.item()
                
                if loss_val < 0 or math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"❌ Invalid loss: {loss_val}")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
        scheduler.step()
        
        # Validation (every epoch)
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for deg, gt, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                deg, gt = deg.to(device), gt.to(device)
                
                if USE_AMP:
                    with autocast():
                        pred = model(deg)
                else:
                    pred = model(deg)
                
                val_psnr += psnr_metric(pred, gt).item()
                val_ssim += ssim_metric(pred, gt).item()
        
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        print(f"📊 Epoch {epoch} | Train Loss: {train_loss/len(mini_train_loader):.4f} | Val PSNR: {avg_val_psnr:.2f} | Val SSIM: {avg_val_ssim:.4f}")
        
        # Save best checkpoint
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': avg_val_ssim,
            }, os.path.join(CHECKPOINT_DIR, 'best_restormer.pth'))
            print(f"💾 Saved best model (PSNR: {best_psnr:.2f})")
        
        # Save periodic checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': avg_val_psnr,
            }, periodic_path)
            print(f"💾 Periodic checkpoint saved: epoch {epoch}")

    print(f"🎉 Training complete! Best PSNR: {best_psnr:.2f}")

if __name__ == '__main__':
    main()
