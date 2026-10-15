from __future__ import annotations

from src.configs.training_config import *

# --------------------------------------------------
# Task 15.4 — Capacity Experiment
# --------------------------------------------------

BASE_CHANNELS = 48

EPOCHS = 3

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "capacity_48"
RESULTS_ROOT = PROJECT_ROOT / "results" / "capacity_48"

SAVE_EVERY_EPOCH = 3