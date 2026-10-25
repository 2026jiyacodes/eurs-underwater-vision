import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/kaggle/input/datasets/jiyachandna/eurs-train/train"
MASK_DIR = os.path.join(DATA_ROOT, "gt_masks")

def calculate_class_weights():
    """
    Calculates inverse frequency weights for each class.
    Rare classes get higher weights to prevent the model from ignoring them.
    """
    print("🔍 Analyzing class distribution in masks...")
    
    class_counts = np.zeros(5)  # 5 classes: 0, 1, 2, 3, 4
    total_pixels = 0
    
    # Get all mask files
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(('.png', '.jpg'))]
    
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Count pixels for each class
        for class_id in range(5):
            class_counts[class_id] += np.sum(mask == class_id)
        
        total_pixels += mask.size
    
    print(f"\n📊 Class Distribution:")
    for i, count in enumerate(class_counts):
        percentage = (count / total_pixels) * 100
        print(f"   Class {i}: {count:>12,} pixels ({percentage:6.2f}%)")
    
    # Calculate inverse frequency weights
    # Formula: weight_i = total_pixels / (num_classes * count_i)
    # This gives higher weights to rare classes
    
    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(0)  # If a class doesn't exist, ignore it
        else:
            weight = total_pixels / (5 * count)
            weights.append(weight)
    
    # Normalize weights so they sum to num_classes
    weights = np.array(weights)
    weights = weights / weights.sum() * 5
    
    print(f"\n⚖️  Calculated Class Weights:")
    for i, weight in enumerate(weights):
        print(f"   Class {i}: {weight:.4f}")
    
    return weights.tolist()

if __name__ == "__main__":
    weights = calculate_class_weights()
    print(f"\n✅ Copy this into train.py:")
    print(f"CLASS_WEIGHTS = {weights}")