import os

# ==============================================================================
# 1. DATA PATHS (UPDATE THIS)
# ==============================================================================
# Local: r"C:\Users\maheit\Downloads\eurs_train\train"
# Kaggle: "/kaggle/input/YOUR_DATASET_NAME/eurs_train/train"
DATA_ROOT = r"C:\Users\maheit\Downloads\eurs_train\train"
DEGRADED_DIR = os.path.join(DATA_ROOT, 'degraded')
GT_DIR = os.path.join(DATA_ROOT, 'gt_images')
CHECKPOINT_DIR = "phase1/checkpoints"

# ==============================================================================
# 2. TRAINING HYPERPARAMETERS (Optimized for 16GB VRAM)
# ==============================================================================
IMG_SIZE = 256          # 256x256 random crops (upgraded from 128)
BATCH_SIZE = 4          # Safe for 16GB with AMP
NUM_EPOCHS = 50         # Proper training duration (upgraded from 10)
LEARNING_RATE = 1e-4
NUM_WORKERS = 2         # Faster data loading

# ==============================================================================
# 3. FULL RESTORMER ARCHITECTURE
# ==============================================================================
RESTORMER_DIM = 48
RESTORMER_NUM_BLOCKS = [4, 6, 6, 8]
RESTORMER_NUM_REFINEMENT_BLOCKS = 4
RESTORMER_HEADS = [1, 2, 4, 8]
