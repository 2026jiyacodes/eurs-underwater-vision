import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models import JointPipeline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5

def preprocess_image(image_path, patch_size=256):
    """Applies the exact same Physics Front-End as the training dataset."""
    img = Image.open(image_path).convert('RGB')
    
    # Center crop or resize to patch_size for inference consistency
    w, h = img.size
    x = (w - patch_size) // 2
    y = (h - patch_size) // 2
    img_cropped = TF.crop(img, y, x, patch_size, patch_size)
    
    # 1. Convert to tensor
    deg_tensor = TF.to_tensor(img_cropped) # [3, H, W]
    
    # 2. Red-Channel Stretch
    r_min, r_max = deg_tensor[0].min(), deg_tensor[0].max()
    deg_tensor[0] = (deg_tensor[0] - r_min) / (r_max - r_min + 1e-6)
    
    # 3. WDCP
    gb_min = deg_tensor[1:].min(dim=0, keepdim=True)[0]
    pad = 15 // 2
    dark_channel = -F.max_pool2d(-gb_min, kernel_size=15, stride=1, padding=pad)
    t_map = (dark_channel - dark_channel.min()) / (dark_channel.max() - dark_channel.min() + 1e-6)
    
    # 4. Concatenate
    input_4ch = torch.cat([deg_tensor, t_map], dim=0).unsqueeze(0) # [1, 4, H, W]
    return input_4ch.to(DEVICE), img_cropped

def main():
    # 1. Load Model
    print("🚀 Loading Model...")
    model = JointPipeline(num_classes=NUM_CLASSES).to(DEVICE)
    
    # TODO: Point this to your trained checkpoint
    checkpoint_path = 'checkpoints/bbest_composite_model.pth' 
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    # 2. Load Image
    image_path = 'test_underwater.jpg' # TODO: Replace with your test image path
    input_4ch, original_img = preprocess_image(image_path)
    
    # 3. Inference
    print("⚡ Running Inference...")
    with torch.no_grad():
        restored_img, seg_logits, _ = model(input_4ch)
        
    # 4. Post-process outputs
    restored_img = restored_img.squeeze(0).clamp(0, 1).cpu()
    seg_preds = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
    
    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Degraded")
    axes[0].axis('off')
    
    axes[1].imshow(restored_img.permute(1, 2, 0))
    axes[1].set_title("Restored (Micro-Restormer)")
    axes[1].axis('off')
    
    # Colorize segmentation mask (assuming 3 classes: 0=bg, 1=class1, 2=class2)
    cmap = plt.get_cmap('tab10', NUM_CLASSES)
    axes[2].imshow(seg_preds, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
    axes[2].set_title("Segmentation (Micro-ViT)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=300)
    print("✅ Result saved as 'inference_result.png'")
    plt.show()

if __name__ == "__main__":
    main()