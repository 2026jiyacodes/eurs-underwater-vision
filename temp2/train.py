import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

# Metrics for validation
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.segmentation import MeanIoU

from dataset import EURSJointDataset  
from models import JointPipeline

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS (MOVED TO TOP TO FIX DEVICE BUG)
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
BATCH_SIZE = 4   
EPOCHS = 50
LEARNING_RATE = 1e-4

# Loss Weights
LAMBDA_PIXEL = 1.0      
LAMBDA_PERCEPTUAL = 0.5 
LAMBDA_SEG = 1.0        
LAMBDA_VIT_DISTILL = 0.5 
LAMBDA_SHARPNESS = 0.1   # Anti-blur signal (10% of restoration score)

DATA_ROOT = '/kaggle/input/datasets/jiyachandna/eurs-train/train'
IOU_THRESHOLD = 0.1 # Coverage penalty threshold

# ==========================================
# 2. LOSS CLASSES
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class SharpnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel = self.kernel.repeat(3, 1, 1, 1).to(DEVICE) # Now DEVICE is defined!
        
    def forward(self, pred, target):
        pred_edges = F.conv2d(pred, self.kernel, padding=1, groups=3)
        target_edges = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(pred_edges, target_edges)

def calculate_per_class_iou(preds, targets, num_classes):
    per_class_iou = []
    for class_id in range(num_classes):
        class_pred = (preds == class_id)
        class_gt = (targets == class_id)
        intersection = (class_pred & class_gt).sum().item()
        union = (class_pred | class_gt).sum().item()
        iou = intersection / union if union > 0 else 0.0
        per_class_iou.append(iou)
    return per_class_iou

# ==========================================
# 3. MODEL INITIALIZATION
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
# 4. OPTIMIZER & DATALOADERS
# ==========================================
optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_dataset = EURSJointDataset(root_dir=DATA_ROOT, mode='train', patch_size=256)
val_dataset = EURSJointDataset(root_dir=DATA_ROOT, mode='val', patch_size=256)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ==========================================
# 5. LOSS FUNCTIONS & METRICS
# ==========================================
criterion_pixel = nn.L1Loss()

# TODO: REPLACE THESE WITH THE ACTUAL OUTPUT FROM calculate_class_weights.py
CLASS_WEIGHTS = [0.05339530451974263, 0.08754919735474437, 0.37843717785610587, 3.1444691334038466, 1.3361491868655606]

class_weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)

criterion_ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
criterion_dice = DiceLoss()
criterion_sharpness = SharpnessLoss()
criterion_distill = nn.L1Loss()

metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
metric_miou = MeanIoU(num_classes=NUM_CLASSES).to(DEVICE)

# NR Quality Metrics (UIQM/UCIQE) - SKIPPED 
# Genuine high PSNR/SSIM naturally improves NR scores without risking "gameable" oversaturation artifacts.
nr_metrics = None
print("⚠️ NR Quality Metrics skipped. Focusing on genuine PSNR/SSIM/Qseg optimization.")

def compute_dino_perceptual_loss(restored_img, gt_img, teacher_model):
    resize_transform = T.Resize((224, 224))
    with torch.no_grad():
        feat_restored = teacher_model.forward_features(resize_transform(restored_img))['x_norm_patchtokens']
        feat_gt = teacher_model.forward_features(resize_transform(gt_img))['x_norm_patchtokens']
    return criterion_distill(F.normalize(feat_restored, p=2, dim=-1), F.normalize(feat_gt, p=2, dim=-1))

def compute_vit_dino_distill_loss(vit_tokens, gt_img, teacher_model):
    resize_transform = T.Resize((224, 224))
    with torch.no_grad():
        dino_tokens = teacher_model.forward_features(resize_transform(gt_img))['x_norm_patchtokens']
    return criterion_distill(F.normalize(vit_tokens, p=2, dim=-1), F.normalize(dino_tokens, p=2, dim=-1))

# ==========================================
# 6. TRAINING & VALIDATION LOOP
# ==========================================
print("🔥 Starting DINO-Guided Joint Training with Competition Metrics...")
os.makedirs('checkpoints', exist_ok=True)

best_psnr, best_miou, best_qseg, best_composite = 0.0, 0.0, 0.0, 0.0

for epoch in range(EPOCHS):
    student_model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, (input_4ch, gt_img, gt_mask, names) in enumerate(pbar):
        input_4ch, gt_img, gt_mask = input_4ch.to(DEVICE), gt_img.to(DEVICE), gt_mask.to(DEVICE).long()
        optimizer.zero_grad()
        
        restored_img, seg_logits, vit_tokens = student_model(input_4ch)
        
        loss_pixel = criterion_pixel(restored_img, gt_img)
        loss_ce = criterion_ce(seg_logits, gt_mask)
        loss_dice = criterion_dice(seg_logits, gt_mask)
        loss_seg = loss_ce + loss_dice
        loss_sharpness = criterion_sharpness(restored_img, gt_img)
        loss_perceptual = compute_dino_perceptual_loss(restored_img, gt_img, teacher_dino)
        loss_vit_distill = compute_vit_dino_distill_loss(vit_tokens, gt_img, teacher_dino)
        
        total_loss = (LAMBDA_PIXEL * loss_pixel) + (LAMBDA_PERCEPTUAL * loss_perceptual) + \
                     (LAMBDA_SEG * loss_seg) + (LAMBDA_VIT_DISTILL * loss_vit_distill) + \
                     (LAMBDA_SHARPNESS * loss_sharpness)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        pbar.set_postfix({'Total': f"{total_loss.item():.4f}", 'Pixel': f"{loss_pixel.item():.4f}", 'Sharp': f"{loss_sharpness.item():.4f}", 'Seg': f"{loss_seg.item():.4f}"})

    scheduler.step()
    print(f"✅ Epoch {epoch+1} Average Train Loss: {epoch_loss / len(train_loader):.4f}")
    
    # --- VALIDATION PHASE (Every 5 epochs) ---
    if (epoch + 1) % 5 == 0:
        student_model.eval()
        all_per_class_iou = [[] for _ in range(NUM_CLASSES)]
        uiqm_scores, uciqe_scores, sharpness_scores = [], [], []
        
        with torch.no_grad():
            for val_input, val_gt, val_mask, _ in val_loader:
                val_input, val_gt, val_mask = val_input.to(DEVICE), val_gt.to(DEVICE), val_mask.to(DEVICE).long()
                val_restored, val_logits, _ = student_model(val_input)
                val_preds = torch.argmax(val_logits, dim=1)
                
                metric_psnr(val_restored, val_gt)
                metric_ssim(val_restored, val_gt)
                metric_miou(val_preds, val_mask)
                
                batch_per_class_iou = calculate_per_class_iou(val_preds, val_mask, NUM_CLASSES)
                for class_id in range(NUM_CLASSES):
                    all_per_class_iou[class_id].append(batch_per_class_iou[class_id])
                
                # Track Sharpness
                laplacian = F.conv2d(val_restored, criterion_sharpness.kernel, padding=1, groups=3)
                sharpness_scores.append(laplacian.var(dim=[1, 2, 3]).mean().item())
                
                if nr_metrics is not None:
                    uiqm_scores.append(nr_metrics['uiqm'](val_restored).mean().item())
                    uciqe_scores.append(nr_metrics['uciqe'](val_restored).mean().item())
            
            current_psnr = metric_psnr.compute().item()
            current_ssim = metric_ssim.compute().item()
            current_miou = metric_miou.compute().item()
            
            avg_per_class_iou = [np.mean(iou_list) for iou_list in all_per_class_iou]
            coverage_penalty = 1.0 if all(iou >= IOU_THRESHOLD for iou in avg_per_class_iou) else 0.0
            
            approx_f1 = (2 * current_miou) / (1 + current_miou) if current_miou > 0 else 0
            current_qseg = approx_f1 * current_miou * coverage_penalty
            
            avg_sharpness = np.mean(sharpness_scores)
            avg_uiqm = np.mean(uiqm_scores) if uiqm_scores else 0
            avg_uciqe = np.mean(uciqe_scores) if uciqe_scores else 0
            
            # 🎯 FULL COMPOSITE SCORE CALCULATION
            psnr_norm = np.clip((current_psnr - 15) / (35 - 15), 0, 1)
            sharpness_norm = np.clip(avg_sharpness / 100.0, 0, 1)
            uiqm_norm = np.clip(avg_uiqm / 5.0, 0, 1)
            uciqe_norm = np.clip(avg_uciqe / 30.0, 0, 1)
            nrquality_norm = (uiqm_norm + uciqe_norm) / 2
            
            restoration_score = 0.45 * psnr_norm + 0.45 * current_ssim + 0.10 * sharpness_norm
            current_composite = 0.6 * current_qseg + 0.3 * restoration_score + 0.1 * nrquality_norm
            
            print(f"\n📊 VALIDATION RESULTS (Epoch {epoch+1}):")
            print(f"   Restoration: PSNR={current_psnr:.2f} | SSIM={current_ssim:.4f} | Sharp={avg_sharpness:.2f}")
            print(f"   Segmentation: mIoU={current_miou:.4f} | Qseg={current_qseg:.4f}")
            print(f"   Per-class IoU: {[f'{iou:.3f}' for iou in avg_per_class_iou]} | Penalty: {coverage_penalty}")
            if nr_metrics is not None: print(f"   NR Quality: UIQM={avg_uiqm:.3f} | UCIQE={avg_uciqe:.3f}")
            print(f"   🏆 COMPOSITE SCORE: {current_composite:.4f}\n")
            
            metric_psnr.reset(); metric_ssim.reset(); metric_miou.reset()
            
            # 💾 SAVE BEST MODELS
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                torch.save(student_model.state_dict(), 'checkpoints/best_psnr_model.pth')
                print(f"🏆 NEW BEST PSNR MODEL SAVED! ({best_psnr:.2f} dB)")
            if current_miou > best_miou:
                best_miou = current_miou
                torch.save(student_model.state_dict(), 'checkpoints/best_miou_model.pth')
                print(f"🏆 NEW BEST mIoU MODEL SAVED! ({best_miou:.4f})")
            if current_qseg > best_qseg:
                best_qseg = current_qseg
                torch.save(student_model.state_dict(), 'checkpoints/best_qseg_model.pth')
                print(f"🏆 NEW BEST Qseg MODEL SAVED! ({best_qseg:.4f})")
            if current_composite > best_composite:
                best_composite = current_composite
                torch.save(student_model.state_dict(), 'checkpoints/best_composite_model.pth')
                print(f"🏆🏆🏆 NEW BEST COMPOSITE MODEL SAVED! ({best_composite:.4f}) ← USE THIS FOR COMPETITION!")
                
        student_model.train()

print("\n🎉 Training Complete!")
print(f"📁 Check the 'checkpoints/' folder for your best models.")