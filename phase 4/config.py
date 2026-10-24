import os

# ==============================================================================
# 1. DATA PATHS (UPDATE THIS FOR KAGGLE)
# ==============================================================================
# Kaggle: "/kaggle/input/YOUR_DATASET_NAME/eurs_train/train"
# Local: r"C:\Users\maheit\Downloads\eurs_train\train"
DATA_ROOT = "/kaggle/input/YOUR_DATASET_NAME/eurs_train/train"
DEGRADED_DIR = os.path.join(DATA_ROOT, 'degraded')
GT_DIR = os.path.join(DATA_ROOT, 'gt_images')
CHECKPOINT_DIR = "phase1/checkpoints"

# ==============================================================================
# 2. TRAINING HYPERPARAMETERS (Optimized for 16GB VRAM)
# ==============================================================================
IMG_SIZE = 256          # 256x256 crops (upgraded from 128 for 16GB)
BATCH_SIZE = 4          # Batch size 4 (upgraded from 2 for 16GB)
NUM_EPOCHS = 50         # Will finish in ~6 hours with mini-epochs
LEARNING_RATE = 5e-5    # Lowered to prevent NaN
NUM_WORKERS = 2         # Can use 2 on Kaggle (faster data loading)

# ==============================================================================
# 3. MINI-EPOCH SETTINGS (Speed Optimization)
# ==============================================================================
IMAGES_PER_EPOCH = 300  # Use 300 images per epoch (instead of all ~1000)
MEMORY_WINDOW = 100     # Remember last 100 images to prevent forgetting

# ==============================================================================
# 4. FULL RESTORMER ARCHITECTURE
# ==============================================================================
RESTORMER_DIM = 48
RESTORMER_NUM_BLOCKS = [4, 6, 6, 8]
RESTORMER_NUM_REFINEMENT_BLOCKS = 4
RESTORMER_HEADS = [1, 2, 4, 8]

# ==============================================================================
# 5. CHECKPOINT SETTINGS
# ==============================================================================
CHECKPOINT_INTERVAL = 10  # Save checkpoint every 10 epochs

# ==============================================================================
# 6. MIXED PRECISION (Use AMP on 16GB for faster training)
# ==============================================================================
USE_AMP = True  # Set to True for 16GB VRAM (faster training)
