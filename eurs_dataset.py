import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

class EURSRestorationDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.deg_dir = os.path.join(root_dir, 'degraded')
        self.gt_dir = os.path.join(root_dir, 'gt_images')
        
        self.samples = []
        # Map every degraded image to its corresponding clean scene image
        for scene in os.listdir(self.deg_dir):
            scene_path = os.path.join(self.deg_dir, scene)
            if os.path.isdir(scene_path):
                gt_img_path = os.path.join(self.gt_dir, f"{scene}.png")
                if os.path.exists(gt_img_path):
                    for deg_img in os.listdir(scene_path):
                        if deg_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append({
                                'deg': os.path.join(scene_path, deg_img),
                                'gt': gt_img_path
                            })
        print(f"✅ Successfully loaded {len(self.samples)} paired training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        deg_img = Image.open(sample['deg']).convert('RGB')
        gt_img = Image.open(sample['gt']).convert('RGB')

        # 1. Random Crop (Crucial for 8GB VRAM)
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

        return deg_tensor, gt_tensor