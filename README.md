# Relative Attributes

+ [Relative Attributes](#relative-attributes)
  + [0. Overview](#0-overview)
  + [1. Features](#1-features)
  + [2. Quick Start](#2-quick-start)
  + [3. Results](#3-results)
    + [3.1. Zappos V1](#31-zappos-v1)
    + [3.2. Zappos V2](#32-zappos-v2)
  + [Reference](#reference)

## 0. Overview

## 1. Features
+ Datasets
  + [x] Zappos50k-1
  + [x] Zappos50k-2
  + [x] LFW10
  + [x] Pubfig
  + [x] Place Pulse
  + [x] OSR
  + [x] Prefetch Dataset(but not working...)
  + [ ] Opencv data argument
  + [ ] Nvidia DALI
  + [ ] Dowloading Scripts
+ Models
  + [x] DRN
  + [ ] DRA
+ Training
  + [x] Logging with `print`
  + [x] Summary with TensorBoard
  + [x] Early Stopping
  + [x] LR decay
  + [ ] Logging with `logging`
  + [ ] Continue Training
+ Scripts
  + [x] Training
  + [ ] generate scoring result files.
+ Examples
  + [x] Datasets notebooks
  + [x] Predicting & Visualization notebooks
    + Cal scores, showing scores hist.
    + Showing images & heatmaps for different graded images.
+ Docs
  + [ ] Quick Start.
  + [ ] Datasets introduction.
  + [ ] Models introduction.
+ Others
  + [ ] GradCam for all models.
  + [ ] More CNN Visualization methods.

## 2. Quick Start

## 3. Results

### 3.1. Zappos V1
|        Model        | Open  | Pointy | Sporty | Comfort | Mean  |             comments              |
| :-----------------: | :---: | :----: | :----: | :-----: | :---: | :-------------------------------: |
|  DRN-VGG16(paper)   | 95.37 | 94.43  | 97.30  |  95.57  | 95.67 |                 /                 |
|   DRN-VGG16(ours)   | 96.33 | 94.33  | 96.33  |  94.00  |   /   | ranknet + lr(1e-4, 1e-5) + wd1e-5 |
|   DRN-VGG16(ours)   | 96.33 | 94.33  | 96.33  |  95.00  |   /   |   dra + lr(1e-4, 1e-5) + wd1e-5   |
| DRN-googlenet(ours) | 92.00 | 94.33  | 94.33  |  95.33  |   /   | ranknet + lr(1e-4, 1e-5) + wd1e-5 |


### 3.2. Zappos V2
|      Model       | Open  | Pointy | Sporty | Comfort | Mean  |                comments                 |
| :--------------: | :---: | :----: | :----: | :-----: | :---: | :-------------------------------------: |
| DRN-VGG16(paper) | 73.45 | 68.20  | 73.07  |  70.31  | 71.26 |                  VGG16                  |
| DRN-VGG16(ours)  | 72.73 | 67.83  | 74.24  |  67.93  |   /   | drn + ranknet + lr(1e-4, 1e-5) + wd1e-5 |
| DRN-VGG16(ours)  | 74.55 | 66.81  | 72.11  |  69.46  |   /   | drn + ranknet + lr(1e-4, 1e-5) + wd5e-5 |


## Reference