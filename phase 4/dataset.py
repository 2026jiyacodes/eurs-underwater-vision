import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

class EURSRestorationDataset(Dataset):
    def __init__(self, root_dir, mode='train', patch_size=256):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode
        
        self.deg_dir = os.path.join(root_dir, 'degraded')
        self.gt_dir = os.path.join(root_dir, 'gt_images')
        
        # Get all scenes (subdirectories in degraded/)
        all_scenes = sorted([
            scene for scene in os.listdir(self.deg_dir)
            if os.path.isdir(os.path.join(self.deg_dir, scene))
        ])
        
        # LEAKAGE-PROOF SPLIT: Split by scene, not by image
        num_val_scenes = max(1, int(len(all_scenes) * 0.10))
        self.val_scenes = all_scenes[-num_val_scenes:]
        self.train_scenes = all_scenes[:-num_val_scenes]
        
        print(f"[*] Total scenes: {len(all_scenes)}")
        print(f"[*] Train scenes ({len(self.train_scenes)}): {self.train_scenes}")
        print(f"[*] Val scenes ({len(self.val_scenes)}): {self.val_scenes}")
        
        # Filter scenes based on mode
        allowed_scenes = self.train_scenes if mode == 'train' else self.val_scenes
        
        self.samples = []
        for scene in allowed_scenes:
            scene_path = os.path.join(self.deg_dir, scene)
            gt_img_path = os.path.join(self.gt_dir, f"{scene}.png")
            
            if os.path.exists(gt_img_path):
                for deg_img in os.listdir(scene_path):
                    if deg_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'deg': os.path.join(scene_path, deg_img),
                            'gt': gt_img_path,
                            'deg_name': f"{scene}_{deg_img}"
                        })
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for mode={mode}. Check your directory structure.")
        
        print(f"✅ {mode.upper()} set: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        deg_img = Image.open(sample['deg']).convert('RGB')
        gt_img = Image.open(sample['gt']).convert('RGB')

        # 1. Random Crop
        w, h = deg_img.size
        ps = min(self.patch_size, w, h)
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        
        deg_img = TF.crop(deg_img, y, x, ps, ps)
        gt_img = TF.crop(gt_img, y, x, ps, ps)

        # 2. Random Augmentations
        if random.random() > 0.5:
            deg_img = TF.hflip(deg_img)
            gt_img = TF.hflip(gt_img)
        if random.random() > 0.5:
            deg_img = TF.vflip(deg_img)
            gt_img = TF.vflip(gt_img)

        # 3. Convert to Tensor [0, 1]
        deg_tensor = TF.to_tensor(deg_img)
        gt_tensor = TF.to_tensor(gt_img)

        return deg_tensor, gt_tensor, sample['deg_name']