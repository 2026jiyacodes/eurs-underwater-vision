import os

# ==============================================================================
# 1. DATA PATHS (UPDATE THIS FOR KAGGLE)
# ==============================================================================
DATA_ROOT = "/kaggle/input/YOUR_DATASET_NAME_HERE/eurs_train/train"
DEGRADED_DIR = os.path.join(DATA_ROOT, 'degraded')
GT_DIR = os.path.join(DATA_ROOT, 'gt_images')
CHECKPOINT_DIR = "phase1/checkpoints"

# ==============================================================================
# 2. TRAINING HYPERPARAMETERS (Optimized for Kaggle 16GB VRAM)
# ==============================================================================
IMG_SIZE = 256          # 256x256 crops
BATCH_SIZE = 4          # Safe for 16GB with Full Restormer. (Drop to 2 if OOM)
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 2         

# ==============================================================================
# 3. FULL RESTORMER ARCHITECTURE
# ==============================================================================
RESTORMER_DIM = 48
RESTORMER_NUM_BLOCKS = [4, 6, 6, 8]
RESTORMER_NUM_REFINEMENT_BLOCKS = 4
RESTORMER_HEADS = [1, 2, 4, 8]