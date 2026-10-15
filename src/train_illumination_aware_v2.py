from __future__ import annotations

import torch

import src.train as train_module

from src.models.illumination_aware_unet_v2 import (
    IlluminationAwareConditionalUNet,
)

from src.configs import (
    illumination_aware_v2_training_config,
)

from src.diffusion.illumination_aware_loss import (
    illumination_aware_training_loss,
)


# ---------------------------------------------------------
# Replace configuration
# ---------------------------------------------------------

train_module.config = illumination_aware_v2_training_config


# ---------------------------------------------------------
# Replace model creation
# ---------------------------------------------------------

def create_model(device: torch.device):
    model = IlluminationAwareConditionalUNet(
        image_channels=(
            illumination_aware_v2_training_config.IMAGE_CHANNELS
        ),
        condition_channels=(
            illumination_aware_v2_training_config.CONDITION_CHANNELS
        ),
        base_channels=(
            illumination_aware_v2_training_config.BASE_CHANNELS
        ),
        time_dim=(
            illumination_aware_v2_training_config.TIME_EMBEDDING_DIM
        ),
    )

    return model.to(device)


train_module.create_model = create_model


# ---------------------------------------------------------
# Replace diffusion training loss
# ---------------------------------------------------------

train_module.diffusion_training_loss = (
    illumination_aware_training_loss
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    train_module.main()