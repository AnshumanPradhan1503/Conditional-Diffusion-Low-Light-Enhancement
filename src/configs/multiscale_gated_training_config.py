from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "LOLv2-real"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "multiscale_gated"
RESULTS_ROOT = PROJECT_ROOT / "results" / "multiscale_gated"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# IMAGE / DATASET SETTINGS
# ============================================================

IMAGE_SIZE = 256

BATCH_SIZE = 2

NUM_WORKERS = 0

PIN_MEMORY = False


# ============================================================
# DIFFUSION SETTINGS
# ============================================================

TIMESTEPS = 1000

BETA_START = 1e-4

BETA_END = 0.02


# ============================================================
# MODEL SETTINGS
# ============================================================

IMAGE_CHANNELS = 3

CONDITION_CHANNELS = 3

BASE_CHANNELS = 32

TIME_EMBEDDING_DIM = 128


# ============================================================
# OPTIMIZATION SETTINGS
# ============================================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EPOCHS = 20

GRADIENT_CLIP_MAX_NORM = 1.0


# ============================================================
# CHECKPOINT / VALIDATION SETTINGS
# ============================================================

SAVE_EVERY_EPOCH = 5

VALIDATE_EVERY_EPOCH = 1


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cpu"