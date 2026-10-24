import os
import sys
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Add Restormer submodule to path
sys.path.append(os.path.abspath('Restormer'))
from basicsr.models.archs.restormer_arch import Restormer

from config import *

def inference(input_dir, output_dir):
    """
    Processes nested folder structure:
    input_dir/scene1/img.png -> output_dir/scene1/img.png
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model (Full Restormer)
    model = Restormer(
        inp_channels=3, out_channels=3, dim=RESTORMER_DIM,
        num_blocks=RESTORMER_NUM_BLOCKS, 
        num_refinement_blocks=RESTORMER_NUM_REFINEMENT_BLOCKS,
        heads=RESTORMER_HEADS,
        ffn_expansion_factor=2.66, bias=False, 
        LayerNorm_type='WithBias', dual_pixel_task=False
    ).to(device)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_restormer.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint not found at {ckpt_path}. Train the model first!")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded model from epoch {checkpoint['epoch']} (PSNR: {checkpoint['psnr']:.2f} dB)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get all scenes (subdirectories)
    scenes = sorted([
        scene for scene in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, scene))
    ])
    
    print(f"[*] Found {len(scenes)} scenes to process")
    total_processed = 0
    
    # 3. Process each scene
    for scene in scenes:
        scene_input_dir = os.path.join(input_dir, scene)
        scene_output_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        img_files = sorted([
            f for f in os.listdir(scene_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        print(f"  Processing scene '{scene}': {len(img_files)} images")
        
        with torch.no_grad():
            for img_file in img_files:
                img_path = os.path.join(scene_input_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                
                # Convert to tensor [0, 1] and add batch dimension
                img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
                
                # Forward pass (Full resolution)
                pred = model(img_tensor)
                
                # Post-process
                pred = torch.clamp(pred, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                pred_img = Image.fromarray((pred * 255).astype('uint8'))
                
                # Save
                save_path = os.path.join(scene_output_dir, img_file)
                pred_img.save(save_path)
                total_processed += 1
            
    print(f"✅ Inference complete. Processed {total_processed} images total.")
    print(f"💾 Outputs saved to {output_dir}")

if __name__ == '__main__':
    # UPDATE THESE PATHS when you are ready to generate outputs!
    # INPUT_DIR = r"C:\Users\maheit\Downloads\eurs_train\train\degraded"
    # OUTPUT_DIR = r"C:\Users\maheit\Downloads\restored_images"
    
    # inference(INPUT_DIR, OUTPUT_DIR)
    pass
