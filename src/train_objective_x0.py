from __future__ import annotations

import torch

import src.train as train_module

from src.configs import objective_x0_config
from src.diffusion.improved_loss import (
    improved_diffusion_training_loss,
)


# ============================================================
# USE EXPERIMENT CONFIGURATION
# ============================================================

train_module.config = objective_x0_config


# ============================================================
# REPLACE ONLY THE LOSS FUNCTION
# ============================================================

def experimental_loss(
    model,
    low_light,
    normal_light,
    alpha_bars,
):
    total_loss, epsilon_loss, x0_loss = (
        improved_diffusion_training_loss(
            model=model,
            low_light=low_light,
            normal_light=normal_light,
            alpha_bars=alpha_bars,
            x0_weight=objective_x0_config.X0_LOSS_WEIGHT,
        )
    )

    return total_loss


train_module.diffusion_training_loss = (
    experimental_loss
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_module.main()