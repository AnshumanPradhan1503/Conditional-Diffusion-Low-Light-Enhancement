# Phase 3 — Preprocessing

## 1. Objective

The objective of Phase 3 was to develop and verify a complete preprocessing pipeline for the LOLv2-real paired low-light image enhancement dataset.

The preprocessing pipeline prepares paired low-light and normal-light images for training a diffusion-based image enhancement model.

---

## 2. Dataset Split

The 689 official training pairs were divided into training and validation subsets using a fixed random seed of 42.

- Training pairs: 551
- Validation pairs: 138
- Official test pairs: 100

The official test set was kept separate from the training and validation data.

---

## 3. Image Resolution

Original dataset images:

- Width: 600 pixels
- Height: 400 pixels
- Channels: RGB

Instead of directly resizing the images to 256×256, paired cropping was selected to avoid aspect-ratio distortion.

Training and validation/test preprocessing produces:

- 256 × 256 image patches

---

## 4. Paired Cropping

Because the dataset contains corresponding low-light and normal-light images, identical spatial operations must be applied to both images.

### Training

Training images use:

- Random 256×256 crop
- Random horizontal flip with probability 0.5

The same crop coordinates and flip operation are applied to both low-light and normal-light images.

### Validation

Validation images use:

- Deterministic center crop
- No random augmentation

### Test

Test images use:

- Deterministic center crop
- No random augmentation

---

## 5. Normalization

Images are converted from uint8 pixel values in the range:

    [0, 255]

to floating-point tensors and normalized to:

    [-1, 1]

The transformation is:

    normalized = pixel / 127.5 - 1

This range is suitable for the planned diffusion model.

---

## 6. Paired Dataset

A custom `LOLv2PairedDataset` class was implemented.

The Dataset:

1. Loads low-light and normal-light image pairs.
2. Selects the appropriate train/validation/test image directory.
3. Applies paired spatial transformations.
4. Converts images to RGB.
5. Converts images to float32 tensors.
6. Normalizes tensors to [-1, 1].
7. Returns low-light and normal-light tensors.

Output tensor shape:

    [3, 256, 256]

---

## 7. DataLoader

PyTorch DataLoaders were implemented with the following initial configuration:

- Batch size: 4
- Number of workers: 0
- Pin memory: False
- Training shuffle: True
- Validation shuffle: False
- Test shuffle: False

Expected batch shape:

    [4, 3, 256, 256]

---

## 8. Verification Results

### Dataset verification

All expected dataset pairs were successfully loaded.

- Training: 551 pairs
- Validation: 138 pairs
- Test: 100 pairs

### Tensor verification

All splits produced:

- Correct tensor shape
- RGB tensors
- float32 dtype
- Values within [-1, 1]
- No NaN values
- No Inf values

### Augmentation verification

Training preprocessing was confirmed to be stochastic.

Validation preprocessing was confirmed to be deterministic.

Test preprocessing was confirmed to be deterministic.

---

## 9. Phase 3 Verification

All nine preprocessing tasks were successfully completed.

| Task | Status |
|------|--------|
| Preprocessing strategy | PASS |
| Train/validation split | PASS |
| Cropping | PASS |
| Normalization | PASS |
| Data augmentation | PASS |
| Paired Dataset | PASS |
| DataLoader | PASS |
| Batch verification | PASS |
| Final preprocessing validation | PASS |

---

## 10. Final Preprocessing Pipeline

    LOLv2-real
        ↓
    Train / Validation / Test split
        ↓
    Paired Dataset
        ↓
    Paired cropping
        ↓
    Training augmentation
        ↓
    RGB conversion
        ↓
    float32 tensor
        ↓
    [-1, 1] normalization
        ↓
    DataLoader
        ↓
    [4, 3, 256, 256] batches
        ↓
    Diffusion model

---

## 11. Conclusion

Phase 3 successfully established a verified preprocessing and data-loading pipeline for the low-light image enhancement task.

The pipeline preserves low-light/normal-light image correspondence, performs appropriate spatial processing, normalizes the data to the required range, and produces valid batches for subsequent diffusion-model training.

Phase 3 is complete.

Next phase:

**Phase 4 — Baseline Diffusion Model**