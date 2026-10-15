from __future__ import annotations

from src.configs.training_config import *


# ============================================================
# TASK 15.5.4 — x0 OBJECTIVE EXPERIMENT
# ============================================================

# Keep the original 32-channel architecture.
BASE_CHANNELS = 32

# Short controlled experiment.
EPOCHS = 3

# x0 reconstruction contribution.
X0_LOSS_WEIGHT = 0.1

# Separate experiment outputs.
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "objective_x0"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "objective_x0"
)

# Save the final checkpoint of this short experiment.
SAVE_EVERY_EPOCH = 3

# Validation every epoch.
VALIDATE_EVERY_EPOCH = 1