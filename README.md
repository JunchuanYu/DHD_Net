<font size='5'>**Dense Hierarchical Dual-Task Network for Landslide Detection in Yunnan Province**</font>

[English](./README.md) | [中文](./README.zh.md)

Haozhuo Huang, [Junchuan Yu](https://github.com/JunchuanYu)☨, Daqing Ge, Shufang Tian, Ling Zhang, Laidian Xi, Qiong Wu, Changhong Hou, Tingyan Fu

☨Corresponding author: yujunchuan@mail.cgs.gov.cn

## Overview

This repository contains the companion source code for the paper and is mainly used to reproduce the proposed DHD-Net model, the three-stage training pipeline, the ablation experiments, and the prediction workflow.

According to the manuscript, the core ideas of DHD-Net are:

- It jointly models multi-source remote sensing data, including InSAR deformation phase maps, DEM, Bing Maps optical imagery, and snow cover information.
- It adopts a dual-task structure to decouple deformation detection and category discrimination, reducing feature interference between tasks.
- It introduces the DFA module in the decoder to alleviate information redundancy caused by the dual-encoder design and improve feature fusion.
- It is designed for large-area landslide hazard identification while balancing detection completeness and category discrimination.

The manuscript reports that DHD-Net achieves the following results on the constructed dataset:

- F1 Score: `77.55%`
- Recall: `80.24%`
- Precision: `75.03%`
- IoU: `63.33%`

The model ultimately identified `4059` landslide hazards across Yunnan Province.

## Repository Structure

```text
DHD_Net/
|-- Model/
|   |-- Ablation_Study/
|   |   |-- DHD_Net_Without_DFA.py
|   |   `-- DHD_Net_Without_DT.py
|   `-- DHD_Net/
|       |-- Classification_Network.py
|       |-- Segmentation_Network.py
|       `-- Dual_Task_Fusion_Network.py
|-- Train.ipynb
|-- Predict.ipynb
|-- Ablation_Study.ipynb
|-- utils.py
|-- README.en.md
|-- README.md
`-- README.zh.md
```

## Method Summary

From the implementation, DHD-Net consists of three parts:

1. Classification network: extracts category-related features from multi-source inputs.
2. Segmentation network: extracts target-region features from deformation-related inputs.
3. Dual-task fusion network: loads and freezes the encoders from the first two stages, then produces final predictions through the fusion decoder.

![](figure.jpg 'DHD-Net')

## Environment

The code is provided in Jupyter Notebook format. The notebook metadata indicates the following main runtime environment:

- Python `3.9`
- PyTorch
- torchvision
- numpy
- pandas
- GDAL
- Jupyter Notebook

You can install the required dependencies according to your local CUDA and Python environment. For code reading or structural reproduction, at least `torch`, `torchvision`, `numpy`, `pandas`, and `gdal` should be available.

## Training

The training entry is `Train.ipynb`, which follows a three-stage training strategy:

### Stage 1: Classification Network

- Train the classification branch with `Classification_Network`.
- Samples whose filename prefix is `0` are skipped by default.
- The optimizer is `AdamW`.
- The default learning rate is `0.001`, and `batch_size` is `8`.
- The monitored metric is `MRecall`.

### Stage 2: Segmentation Network

- Train the segmentation branch with `Segmentation_Network`.
- The default number of segmentation classes is `2`.
- The monitored metric is `Binary IoU`.

### Stage 3: Dual-Task Fusion Network

- Build `Dual_Task_Fusion_Network`.
- Load encoder weights from the first two stages.
- Freeze the parameters of `stageMS_*` and `stageSAR_*`.
- Train the fusion decoder and save the final weights to `./Model_Weight/DHD-Net.pkl`.
- The monitored metric is `F1_1`.

The default hyperparameters in the notebook are:

```python
class_num = 6
seg_class_num = 2
lr = 0.001
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.01
class_epochs = 150
seg_epochs = 150
dual_epochs = 150
batch_size = 8
```

The training pipeline also implements early stopping based on a moving average over a sliding window.

## Inference

The prediction entry is `Predict.ipynb`.

The default workflow is:

- Read `.tif` files from `--data_dir`.
- Run inference tile by tile with a sliding-window strategy.
- Use a default window size of `img_size = 128` and a stride of `stride = 128`.
- Save outputs to `./Data/Predictions`.
- Preserve the original GeoTIFF projection and georeferencing information in the output files.

Default weight path:

```text
./Model_Weight/DHD-Net.pkl
```

## Ablation Study

The ablation entry is `Ablation_Study.ipynb`. The current repository includes two core ablation settings from the manuscript:

- `Without Dual-Task`
- `Without DFA`

The test-set comparison reported in the notebook is:

| Model | F1 Score (%) | Recall (%) | Precision (%) | IoU (%) |
| --- | ---: | ---: | ---: | ---: |
| UNet++ | 70.33 | 70.85 | 69.82 | 54.24 |
| Without Dual-Task | 75.17 | 73.23 | 77.21 | 60.21 |
| Without DFA | 75.25 | 75.51 | 75.00 | 60.32 |
| DHD-Net | 77.55 | 80.24 | 75.03 | 63.33 |

## Citation

If this repository is helpful to your research, please prioritize citing the formally published version. For now, you may refer to the manuscript authors:

```text
Haozhuo Huang, Junchuan Yu, Daqing Ge, Shufang Tian, Ling Zhang,
Laidian Xi, Qiong Wu, Changhong Hou, Tingyan Fu.
Manuscript.
```

