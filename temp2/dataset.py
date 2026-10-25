import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

IMG_EXTS = ('.png', '.jpg', '.jpeg')

class EURSJointDataset(Dataset):
    def __init__(self, root_dir, mode='train', patch_size=256, num_val_scenes=1):
        assert mode in ('train', 'val'), f"bad mode: {mode}"
        self.patch_size = patch_size
        self.mode = mode
        self.deg_dir = os.path.join(root_dir, 'degraded')
        self.gt_dir = os.path.join(root_dir, 'gt_images')
        self.mask_dir = os.path.join(root_dir, 'gt_masks') # <-- ADDED MASK DIR
        self._gt_cache = {}
        self._mask_cache = {}

        all_scenes = sorted(s for s in os.listdir(self.deg_dir) if os.path.isdir(os.path.join(self.deg_dir, s)))
        assert len(all_scenes) > num_val_scenes, "not enough scenes"
        self.val_scenes = all_scenes[-num_val_scenes:]
        self.train_scenes = all_scenes[:-num_val_scenes]

        allowed = self.train_scenes if mode == 'train' else self.val_scenes
        print(f"[*] {mode}: scenes {allowed}")

        self.samples = []
        for scene in allowed:
            gt_path = os.path.join(self.gt_dir, f"{scene}.png")
            mask_path = os.path.join(self.mask_dir, f"{scene}.png") # Assumes 1 mask per scene. Change to f"{scene}_{name}" if per-image.
            
            if not os.path.exists(gt_path) or not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing GT or Mask for scene '{scene}'")
                
            scene_path = os.path.join(self.deg_dir, scene)
            for name in sorted(os.listdir(scene_path)):
                if name.lower().endswith(IMG_EXTS):
                    self.samples.append({
                        'deg': os.path.join(scene_path, name),
                        'gt': gt_path,
                        'mask': mask_path,
                        'deg_name': f"{scene}_{name}",
                    })

        print(f"✅ {mode.upper()}: {len(self.samples)} samples")

    def _load_image(self, path, cache_dict):
        if path not in cache_dict:
            with Image.open(path) as im:
                cache_dict[path] = im.convert('RGB')
        return cache_dict[path]

    def _load_mask(self, path, cache_dict):
        if path not in cache_dict:
            with Image.open(path) as im:
                # Masks should be loaded as 'P' (palette) or 'L' (grayscale) to preserve class indices
                cache_dict[path] = im.convert('L') 
        return cache_dict[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        deg = self._load_image(s['deg'], {}) # Don't cache deg, it's unique per file
        gt = self._load_image(s['gt'], self._gt_cache)
        mask = self._load_mask(s['mask'], self._mask_cache)

        w, h = deg.size
        ps = self.patch_size
        
        if self.mode == 'train':
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
        else:
            x = (w - ps) // 2
            y = (h - ps) // 2

        deg = TF.crop(deg, y, x, ps, ps)
        gt = TF.crop(gt, y, x, ps, ps)
        mask = TF.crop(mask, y, x, ps, ps)

        if self.mode == 'train':
            if random.random() > 0.5:
                deg, gt, mask = TF.hflip(deg), TF.hflip(gt), TF.hflip(mask)
            if random.random() > 0.5:
                deg, gt, mask = TF.vflip(deg), TF.vflip(gt), TF.vflip(mask)

        # 1. Convert to tensors
        deg_tensor = TF.to_tensor(deg) # [3, H, W], scaled 0-1
        gt_tensor = TF.to_tensor(gt)   # [3, H, W], scaled 0-1
        mask_tensor = TF.pil_to_tensor(mask).squeeze(0).long() # [H, W], integer class labels (DO NOT scale to 0-1)

        # 2. Red-Channel Stretch
        r_min, r_max = deg_tensor[0].min(), deg_tensor[0].max()
        deg_tensor[0] = (deg_tensor[0] - r_min) / (r_max - r_min + 1e-6)

        # 3. Water-Aware Dark Channel Prior (WDCP)
        gb_img = deg_tensor[1:, :, :].unsqueeze(0) # [1, 2, H, W]
        dark_channel = gb_img.min(dim=1, keepdim=True)[0] # [1, 1, H, W]
        pad = 15 // 2
        dark_channel = -F.max_pool2d(-dark_channel, kernel_size=15, stride=1, padding=pad)
        t_map = (dark_channel - dark_channel.min()) / (dark_channel.max() - dark_channel.min() + 1e-6)
        t_map = t_map.squeeze(0) # [1, H, W]

        # 4. Concatenate for 4-channel input
        input_4ch = torch.cat([deg_tensor, t_map], dim=0) # [4, H, W]

        return input_4ch, gt_tensor, mask_tensor, s['deg_name']