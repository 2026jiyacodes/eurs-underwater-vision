DATA_ROOT = "/kaggle/input/<actual-dataset-slug>/eurs_train/train"
DEGRADED_DIR = os.path.join(DATA_ROOT, 'degraded')
GT_DIR = os.path.join(DATA_ROOT, 'gt_images')
CHECKPOINT_DIR = "/kaggle/working/phase4/checkpoints"

IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
NUM_WORKERS = 2
SEED = 42

IMAGES_PER_EPOCH = 300
MEMORY_WINDOW = 100

USE_SPECTRAL_PENALTY = True
SPECTRAL_PENALTY_WEIGHT = 0.01
USE_AMP = True
CHECKPOINT_INTERVAL = 10
