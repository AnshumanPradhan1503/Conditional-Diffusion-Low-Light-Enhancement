from __future__ import annotations

import src.train as train_module

from src.configs import multiscale_gated_training_config
from src.models.multiscale_gated_unet import MultiScaleGatedConditionalUNet


# ============================================================
# USE THE GATED-MULTISCALE CONFIGURATION
# ============================================================

train_module.config = multiscale_gated_training_config


# ============================================================
# CREATE GATED MULTI-SCALE MODEL
# ============================================================

def create_multiscale_gated_model(device):
    model = MultiScaleGatedConditionalUNet(
        image_channels=train_module.config.IMAGE_CHANNELS,
        condition_channels=train_module.config.CONDITION_CHANNELS,
        base_channels=train_module.config.BASE_CHANNELS,
        time_dim=train_module.config.TIME_EMBEDDING_DIM,
    )

    model = model.to(device)

    return model


# Replace the model factory used by src.train
train_module.create_model = create_multiscale_gated_model


# ============================================================
# TRAINING ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_module.main()