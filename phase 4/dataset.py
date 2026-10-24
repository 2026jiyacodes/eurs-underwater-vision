import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

IMG_EXTS = ('.png', '.jpg', '.jpeg')


class EURSRestorationDataset(Dataset):
    def __init__(self, root_dir, mode='train', patch_size=256, num_val_scenes=1):
        assert mode in ('train', 'val'), f"bad mode: {mode}"
        self.patch_size = patch_size
        self.mode = mode
        self.deg_dir = os.path.join(root_dir, 'degraded')
        self.gt_dir = os.path.join(root_dir, 'gt_images')
        self._gt_cache = {}

        all_scenes = sorted(
            s for s in os.listdir(self.deg_dir)
            if os.path.isdir(os.path.join(self.deg_dir, s))
        )
        assert len(all_scenes) > num_val_scenes, "not enough scenes"
        self.val_scenes = all_scenes[-num_val_scenes:]
        self.train_scenes = all_scenes[:-num_val_scenes]
        assert not set(self.val_scenes) & set(self.train_scenes)

        allowed = self.train_scenes if mode == 'train' else self.val_scenes
        print(f"[*] {mode}: scenes {allowed}")

        self.samples = []
        for scene in allowed:
            gt_path = os.path.join(self.gt_dir, f"{scene}.png")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Missing GT for scene '{scene}': {gt_path}")
            scene_path = os.path.join(self.deg_dir, scene)
            for name in sorted(os.listdir(scene_path)):
                if name.lower().endswith(IMG_EXTS):
                    self.samples.append({
                        'deg': os.path.join(scene_path, name),
                        'gt': gt_path,
                        'deg_name': f"{scene}_{name}",
                    })

        if not self.samples:
            raise ValueError(f"No samples for mode={mode}")
        print(f"✅ {mode.upper()}: {len(self.samples)} samples")

    def _load_gt(self, path):
        if path not in self._gt_cache:  # cached per worker
            with Image.open(path) as im:
                self._gt_cache[path] = im.convert('RGB')
        return self._gt_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with Image.open(s['deg']) as im:
            deg = im.convert('RGB')
        gt = self._load_gt(s['gt'])

        assert deg.size == gt.size, f"size mismatch {s['deg_name']}: {deg.size} vs {gt.size}"
        w, h = deg.size
        ps = self.patch_size
        assert w >= ps and h >= ps, f"{s['deg_name']} smaller than patch {ps}"

        if self.mode == 'train':
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
        else:  # deterministic center crop
            x = (w - ps) // 2
            y = (h - ps) // 2

        deg = TF.crop(deg, y, x, ps, ps)
        gt = TF.crop(gt, y, x, ps, ps)

        if self.mode == 'train':
            if random.random() > 0.5:
                deg, gt = TF.hflip(deg), TF.hflip(gt)
            if random.random() > 0.5:
                deg, gt = TF.vflip(deg), TF.vflip(gt)

        return TF.to_tensor(deg), TF.to_tensor(gt), s['deg_name']
