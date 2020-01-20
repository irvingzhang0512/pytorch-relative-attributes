# Relative Attributes

+ [Relative Attributes](#relative-attributes)
  + [0. Overview](#0-overview)
  + [1. Features](#1-features)
  + [2. Results](#2-results)
    + [2.1. Zappos V1](#21-zappos-v1)
    + [2.2. Zappos V2](#22-zappos-v2)
  + [Reference](#reference)

## 0. Overview

## 1. Features
+ Datasets
  + [x] Zappos50k-1
  + [x] Zappos50k-2
  + [ ] Pubfig
  + [ ] LFW10
+ Models
  + [x] DRN
  + [ ] DRA
+ Scripts
  + [x] Training
  + [ ] Evaluating
  + [ ] Predicting
+ Examples
  + [x] Show datasets.
  + [ ] Show Models.


## 2. Results

### 2.1. Zappos V1
|     Model      | Open  | Pointy | Sporty | Comfort | Mean  |                    comments                     |
| :------------: | :---: | :----: | :----: | :-----: | :---: | :---------------------------------------------: |
|   DRN(paper)   | 95.37 | 94.43  | 97.30  |  95.57  | 95.67 |                      VGG16                      |
| DRN(this repo) | 96.33 |   /    |   /    |    /    |   /   | VGG16 + drn + ranknet + lr(1e-4, 1e-5) + wd1e-5 |


### 2.2. Zappos V2
|   Model    | Open  | Pointy | Sporty | Comfort | Mean  | comments |
| :--------: | :---: | :----: | :----: | :-----: | :---: | :------: |
| DRN(paper) | 73.45 | 68.20  | 73.07  |  70.31  | 71.26 |  VGG16   |
| DRN(paper) | 72.09 |   /    |   /    |    /    |   /   |  VGG16   |

## Reference