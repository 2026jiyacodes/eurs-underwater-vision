import os
import sys
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Add Restormer to path
sys.path.append(os.path.abspath('Restormer'))
from basicsr.models.archs.restormer_arch import Restormer

# --- Configuration ---
MODEL_PATH = "checkpoints/best_restormer.pth"
# Pick any degraded image from your dataset to test
TEST_IMAGE_PATH = r"C:\Users\maheit\Downloads\eurs_train\train\degraded\fish1\fish1_10_10.png"
OUTPUT_PATH = "restored_output.png"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # 1. Initialize Model
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
    ).to(device)

    # 2. Load Checkpoint
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")
    else:
        print("⚠️ No checkpoint found. Running with random weights (will look like noise). Train first!")

    model.eval()

    # 3. Load and Preprocess Image
    img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    original_size = img.size
    
    # Restormer prefers dimensions divisible by 8. We'll resize for this quick test.
    # For full-resolution inference later, we would use padding.
    target_size = (256, 256) 
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    print(f"📥 Input shape: {img_tensor.shape}")

    # 4. Inference
    with torch.no_grad():
        restored_tensor = model(img_tensor)
    
    # 5. Save Result
    restored_img = TF.to_pil_image(restored_tensor.squeeze(0).clamp(0, 1))
    restored_img.save(OUTPUT_PATH)
    print(f"💾 Restored image saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()