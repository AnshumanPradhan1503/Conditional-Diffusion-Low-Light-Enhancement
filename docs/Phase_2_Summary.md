Phase 2 Summary — Dataset Acquisition, Verification & EDA
Project

Diffusion Models for Low-Light Enhancement

Problem Statement:
Develop a Denoising Diffusion Probabilistic Model (DDPM) pipeline to restore clarity and detail in landscape/low-light photographs.

1. Dataset Selected

We selected the LOLv2-real paired low-light image enhancement dataset.

Dataset structure:

data/
└── LOLv2-real/
    ├── train/
    │   ├── low/
    │   ├── high/
    │   └── metadata.jsonl
    │
    └── test/
        ├── low/
        ├── high/
        └── metadata.jsonl

The dataset provides corresponding:

Low-light image  ↔  Normal-light reference

This paired structure is highly suitable for supervised low-light enhancement.

2. Dataset Size
Split	Low-light	Normal-light	Pairs
Train	689	689	689
Test	100	100	100
Total	789	789	789

Total images:

1,578 images

Total paired samples:

789 pairs

3. Dataset Integrity Verification

We verified:

Image dimensions
600 × 400
Image format
RGB
3 channels
Pairing

The images are correctly paired by corresponding filenames/metadata.

Example:

low00001.png
      ↕
normal00001.png
Corruption
Training corrupted pairs = 0
Testing corrupted pairs  = 0

Therefore:

No corrupted image pairs were detected.

4. Metadata Verification

Both training and testing directories contain:

metadata.jsonl

The metadata specifies the corresponding low-light and normal-light images.

Example:

low_file_name  → Low\low00001.png
high_file_name → Normal\normal00001.png

Our actual directory structure uses:

train/low/
train/high/
test/low/
test/high/

The images themselves were successfully verified against the dataset structure.

5. Visual EDA

We generated representative low-light/normal-light image pairs.

The visual inspection showed that low-light images suffer from:

Severe underexposure
Loss of visible details
Reduced contrast
Reduced color visibility
Very dark shadow regions
Difficult-to-see objects and textures

The corresponding normal-light images reveal substantially more scene information.

Therefore, the task is not simply brightness adjustment.

The model needs to learn:

Low-light image
       ↓
Illumination restoration
       +
Detail recovery
       +
Color/contrast restoration
       ↓
Normal-light image
6. Brightness Analysis
Training set
Metric	Low-light	Normal-light
Mean brightness	16.08	113.88
Std	11.16	26.73
Minimum	0.38	27.53
Maximum	82.59	192.21
Test set
Metric	Low-light	Normal-light
Mean brightness	14.50	94.51
Std	7.78	19.37
Minimum	3.42	58.33
Maximum	36.33	147.12

The large difference confirms substantial illumination degradation.

7. Extremely Dark Pixels

We measured pixels with brightness below 10.

Training
Low-light       → 43.79%
Normal-light    → 1.44%
Testing
Low-light       → 46.27%
Normal-light    → 0.07%

This is a significant finding.

Approximately 44–46% of pixels in the low-light images are extremely dark, while very few pixels in their normal-light counterparts fall into this range.

This demonstrates that the dataset contains genuinely challenging low-light conditions.

8. RGB Analysis
Training

Low-light:

R = 17.87
G = 15.42
B = 14.77

Normal-light:

R = 119.46
G = 112.35
B = 107.13

RGB standard deviation:

Low-light:
R = 12.69
G = 10.62
B = 11.01

Normal-light:
R = 49.33
G = 49.17
B = 48.33

The normal-light images exhibit substantially greater pixel-level variation, indicating stronger tonal and contrast information.

9. Test Set Difficulty

An interesting observation was:

Training low-light mean = 16.08
Testing low-light mean  = 14.50

The test low-light images are, on average, darker than the training low-light images.

This makes the test set useful for evaluating whether the model can generalize to difficult low-light conditions.

10. Darkest Training Samples

The darkest training examples included:

low00148.png → 0.38
low00178.png → 0.70
low00192.png → 1.50
low00256.png → 1.75
low00182.png → 1.90

These examples demonstrate that some inputs are extremely underexposed.

11. Largest Brightness Improvements

Some training pairs showed very large differences between the low-light and normal-light images.

Examples:

low00094.png → 166.51
low00109.png → 164.69
low00015.png → 164.48
low00114.png → 163.69

This indicates that the enhancement model must handle a wide range of illumination degradation.

12. Important Dataset Limitation

The LOLv2-real dataset is not exclusively composed of landscape photographs.

Our visual inspection showed:

Indoor scenes
Objects
Buildings
Corridors
Outdoor scenes
Other environments

Therefore, we will use LOLv2-real as the main training and paired evaluation dataset, while later we can perform an additional landscape-focused generalization experiment.

We should not contaminate the training set with external landscape images.

13. Final Phase 2 Conclusion
Dataset suitability: ✅ YES

The dataset satisfies the requirements for our initial diffusion-model experiment:

Requirement	Result
Paired low/high images	✅
Sufficient training pairs	✅
Consistent resolution	✅
RGB images	✅
Correct pairing	✅
No corrupted pairs	✅
Significant low-light degradation	✅
Challenging dark regions	✅
Separate test set	✅
Suitable for supervised enhancement	✅
Phase 2 Deliverables

We have created:

IBM DL Project/
│
├── data/
│   └── LOLv2-real/
│
├── results/
│   ├── train_image_pairs.png
│   ├── train_brightness_distribution.png
│   ├── test_image_pairs.png
│   ├── test_brightness_distribution.png
│   ├── train_statistical_analysis.csv
│   └── test_statistical_analysis.csv
│
├── analyze_dataset.py
├── eda_images.py
└── statistical_eda.py

These results will also be useful later when writing the project report, methodology, and experimental analysis.

Phase 2 Status
PHASE 2 — DATASET & EDA
════════════════════════════════════

Dataset selection             ✅
Dataset acquisition           ✅
Folder verification           ✅
Image count verification      ✅
Pair verification             ✅
Metadata verification         ✅
Corruption check              ✅
Resolution analysis           ✅
Visual EDA                    ✅
Brightness analysis           ✅
RGB analysis                  ✅
Dark-pixel analysis           ✅
Statistical analysis          ✅

        PHASE 2 COMPLETE
                ↓
        PHASE 3 — PREPROCESSING
Final conclusion

LOLv2-real is a valid and challenging dataset for developing our paired low-light image enhancement pipeline. The EDA demonstrates severe illumination degradation, substantial loss of visible information in dark regions, and significant differences in brightness and tonal variation between low-light inputs and normal-light references.