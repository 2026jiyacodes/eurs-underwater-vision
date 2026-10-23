import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from eurs_dataset import EURSRestorationDataset

# Add Restormer to path
sys.path.append(os.path.abspath('Restormer'))
from basicsr.models.archs.restormer_arch import Restormer

# --- QUICK TEST CONFIGURATION ---
DATA_ROOT = r"C:\Users\maheit\Downloads\eurs_train\train"
PATCH_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("🚀 STARTING QUICK TRAINING TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # 1. Dataset
    print("\n📂 Loading dataset...")
    full_dataset = EURSRestorationDataset(root_dir=DATA_ROOT, patch_size=PATCH_SIZE)
    full_dataset.samples = full_dataset.samples[:200]
    
    val_size = 20
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✅ Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # 2. Model
    print("\n🔧 Loading Restormer model...")
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
    ).to(device)
    print("✅ Model loaded")

    # 3. Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float('inf')

    # 4. Training Loop with progress bars
    print("\n" + "=" * 60)
    print("🏋️ STARTING TRAINING")
    print("=" * 60)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True)
        
        for deg, gt in pbar:
            deg, gt = deg.to(device), gt.to(device)
            
            optimizer.zero_grad()
            output = model(deg)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=True)
        
        with torch.no_grad():
            for deg, gt in pbar_val:
                deg, gt = deg.to(device), gt.to(device)
                output = model(deg)
                val_loss += criterion(output, gt).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_restormer.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"💾 Saved best model to {checkpoint_path}")

    print("\n" + "=" * 60)
    print("🎉 QUICK TEST COMPLETE!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}/best_restormer.pth")

if __name__ == "__main__":
    main()