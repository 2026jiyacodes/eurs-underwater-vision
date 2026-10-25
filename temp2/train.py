import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F

# Metrics for validation
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.segmentation import MeanIoU

from dataset import EURSJointDataset  
from models import JointPipeline

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3
BATCH_SIZE = 8    
EPOCHS = 50
LEARNING_RATE = 1e-4

# Loss Weights
LAMBDA_PIXEL = 1.0      
LAMBDA_PERCEPTUAL = 0.5 
LAMBDA_SEG = 1.0        
LAMBDA_VIT_DISTILL = 0.5 

DATA_ROOT = '' # TODO: Point to your dataset root (e.g., './data/EURS')

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print("🚀 Initializing Student Model...")
student_model = JointPipeline(num_classes=NUM_CLASSES).to(DEVICE)

print("🧊 Initializing Frozen DINO Teacher...")
teacher_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
teacher_dino.eval()
for param in teacher_dino.parameters():
    param.requires_grad = False

print("✅ Models loaded successfully.")

# ==========================================
# 3. OPTIMIZER & DATALOADERS
# ==========================================
optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_dataset = EURSJointDataset(root_dir=DATA_ROOT, mode='train', patch_size=256)
val_dataset = EURSJointDataset(root_dir=DATA_ROOT, mode='val', patch_size=256)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 4. LOSS FUNCTIONS & METRICS
# ==========================================
criterion_pixel = nn.L1Loss()
criterion_seg = nn.CrossEntropyLoss()
criterion_distill = nn.L1Loss()

# Validation Metrics
metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
metric_miou = MeanIoU(num_classes=NUM_CLASSES).to(DEVICE)

def compute_dino_perceptual_loss(restored_img, gt_img, teacher_model):
    resize_transform = T.Resize((224, 224))
    with torch.no_grad():
        feat_restored = teacher_model.forward_features(resize_transform(restored_img))['x_norm_patchtokens']
        feat_gt = teacher_model.forward_features(resize_transform(gt_img))['x_norm_patchtokens']
    
    # L2 Normalize along the feature dimension to prevent magnitude explosion
    return criterion_distill(F.normalize(feat_restored, p=2, dim=-1), 
                             F.normalize(feat_gt, p=2, dim=-1))

def compute_vit_dino_distill_loss(vit_tokens, gt_img, teacher_model):
    resize_transform = T.Resize((224, 224))
    with torch.no_grad():
        dino_tokens = teacher_model.forward_features(resize_transform(gt_img))['x_norm_patchtokens']
    
    return criterion_distill(F.normalize(vit_tokens, p=2, dim=-1),
                             F.normalize(dino_tokens, p=2, dim=-1))

# ==========================================
# 5. TRAINING & VALIDATION LOOP
# ==========================================
print("🔥 Starting DINO-Guided Joint Training...")
best_miou = 0.0

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    student_model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, (input_4ch, gt_img, gt_mask, names) in enumerate(pbar):
        input_4ch = input_4ch.to(DEVICE)
        gt_img = gt_img.to(DEVICE)
        gt_mask = gt_mask.to(DEVICE).long()
        
        optimizer.zero_grad()
        
        # Forward Pass (returns 3 values now)
        restored_img, seg_logits, vit_tokens = student_model(input_4ch)
        
        # Calculate Losses
        loss_pixel = criterion_pixel(restored_img, gt_img)
        loss_seg = criterion_seg(seg_logits, gt_mask)
        loss_perceptual = compute_dino_perceptual_loss(restored_img, gt_img, teacher_dino)
        loss_vit_distill = compute_vit_dino_distill_loss(vit_tokens, gt_img, teacher_dino)
        
        total_loss = (LAMBDA_PIXEL * loss_pixel) + \
                     (LAMBDA_PERCEPTUAL * loss_perceptual) + \
                     (LAMBDA_SEG * loss_seg) + \
                     (LAMBDA_VIT_DISTILL * loss_vit_distill)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        pbar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Pixel': f"{loss_pixel.item():.4f}", 
            'DINO-P': f"{loss_perceptual.item():.4f}",
            'ViT-D': f"{loss_vit_distill.item():.4f}",
            'Seg': f"{loss_seg.item():.4f}"
        })

    scheduler.step()
    print(f"✅ Epoch {epoch+1} Average Train Loss: {epoch_loss / len(train_loader):.4f}")
    
    # --- VALIDATION PHASE (Every 5 epochs) ---
    if (epoch + 1) % 5 == 0:
        student_model.eval()
        with torch.no_grad():
            for val_input, val_gt, val_mask, _ in val_loader:
                val_input = val_input.to(DEVICE)
                val_gt = val_gt.to(DEVICE)
                val_mask = val_mask.to(DEVICE).long()
                
                # Forward pass (ignore vit_tokens during val)
                val_restored, val_logits, _ = student_model(val_input)
                val_preds = torch.argmax(val_logits, dim=1)
                
                # Update metrics
                metric_psnr(val_restored, val_gt)
                metric_ssim(val_restored, val_gt)
                metric_miou(val_preds, val_mask)
                
            # Compute and print final metrics for this epoch
            current_psnr = metric_psnr.compute().item()
            current_ssim = metric_ssim.compute().item()
            current_miou = metric_miou.compute().item()
            
            print(f"📊 VAL -> PSNR: {current_psnr:.2f} | SSIM: {current_ssim:.4f} | mIoU: {current_miou:.4f}")
            
            # Reset metrics for next epoch
            metric_psnr.reset()
            metric_ssim.reset()
            metric_miou.reset()
            
            # SAVE BEST MODEL based on mIoU
            if current_miou > best_miou:
                best_miou = current_miou
                torch.save(student_model.state_dict(), 'best_model.pth')
                print(f"🏆 NEW BEST MODEL SAVED! (mIoU: {best_miou:.4f})")
                
        student_model.train() # Switch back to training mode

print("🎉 Training Complete! Best model saved as 'best_model.pth'")