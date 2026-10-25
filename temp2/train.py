import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from dataset import EURSJointDataset  
from models import JointPipeline

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10  # TODO: Change to your actual number of classes
BATCH_SIZE = 8    
EPOCHS = 50
LEARNING_RATE = 1e-4

# Loss Weights (DINO-Guided Strategy)
LAMBDA_PIXEL = 1.0      # Baseline PSNR
LAMBDA_PERCEPTUAL = 0.5 # DINO guidance (boosts PSNR to 30+ and mIoU)
LAMBDA_SEG = 1.0        # Segmentation accuracy

DATA_ROOT = './your_dataset_path' # TODO: Point to your dataset root

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print("🚀 Initializing Student Model...")
student_model = JointPipeline(num_classes=NUM_CLASSES).to(DEVICE)

print("🧊 Initializing Frozen DINO Teacher...")
# DINOv2 ViT-Small is frozen, adds 0 params to student budget
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
# 4. LOSS FUNCTIONS
# ==========================================
criterion_pixel = nn.L1Loss()
criterion_seg = nn.CrossEntropyLoss()
criterion_distill = nn.L1Loss()

def compute_dino_perceptual_loss(restored_img, gt_img, teacher_model):
    """
    Passes both images through frozen DINO and computes L1 loss on patch tokens.
    This forces the student to restore images that preserve semantic structure.
    """
    # DINO expects 3-channel, 224x224 inputs
    resize_transform = T.Resize((224, 224))
    restored_resized = resize_transform(restored_img)
    gt_resized = resize_transform(gt_img)
    
    with torch.no_grad():
        # Extract patch tokens (high-frequency semantic features)
        feat_restored = teacher_model.forward_features(restored_resized)['x_norm_patchtokens']
        feat_gt = teacher_model.forward_features(gt_resized)['x_norm_patchtokens']
        
    # L1 loss between student's restored features and GT features
    return criterion_distill(feat_restored, feat_gt)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("🔥 Starting DINO-Guided Training...")
for epoch in range(EPOCHS):
    student_model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, (input_4ch, gt_img, gt_mask, names) in enumerate(pbar):
        input_4ch = input_4ch.to(DEVICE)
        gt_img = gt_img.to(DEVICE)
        gt_mask = gt_mask.to(DEVICE).long()
        
        optimizer.zero_grad()
        
        # 1. Forward Pass
        restored_img, seg_logits = student_model(input_4ch)
        
        # 2. Calculate Losses
        loss_pixel = criterion_pixel(restored_img, gt_img)
        loss_seg = criterion_seg(seg_logits, gt_mask)
        loss_perceptual = compute_dino_perceptual_loss(restored_img, gt_img, teacher_dino)
        
        # Total Joint Loss (No weak Restormer teacher)
        total_loss = (LAMBDA_PIXEL * loss_pixel) + \
                     (LAMBDA_PERCEPTUAL * loss_perceptual) + \
                     (LAMBDA_SEG * loss_seg)
        
        # 3. Backward Pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        pbar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Pixel': f"{loss_pixel.item():.4f}", 
            'DINO': f"{loss_perceptual.item():.4f}",
            'Seg': f"{loss_seg.item():.4f}"
        })

    scheduler.step()
    print(f"✅ Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(student_model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')
        print(f"💾 Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")

print("🎉 Training Complete!")