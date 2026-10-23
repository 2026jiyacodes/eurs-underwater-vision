from pathlib import Path

root = Path(r"C:\Users\maheit\Downloads\eurs_train\train")

# 1. Check clean images
gt_imgs = sorted((root / "gt_images").glob("*"))[:3]
print("--- Clean Images (gt_images) ---")
for p in gt_imgs:
    print(f"  {p.name}")

# 2. Check degraded images (inside the first scene folder)
deg_scenes = sorted((root / "degraded").iterdir())
if deg_scenes:
    first_scene = deg_scenes[0]
    deg_imgs = sorted(first_scene.glob("*"))[:3]
    print(f"\n--- Degraded Images (inside '{first_scene.name}') ---")
    for p in deg_imgs:
        print(f"  {p.name}")