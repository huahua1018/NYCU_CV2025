# NYCU Computer Vision 2025 Spring HW3
StudentID: 313551093\
Name: 盧品樺

## Introduction

In this assignment, we need to use **Mask R-CNN** to perform an **instance segmentation** task, with the goal of
segmenting four types of cells.


Faster R-CNN consists of three main components: a backbone, a neck (Region Proposal Network, RPN), and a head.

Mask R-CNN is an extension of Faster R-CNN, which introduces an additional branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branches for classification and bounding box regression.

Similar to the previous assignment, the architecture can be divided into three main components: a back-
bone, a neck (Region Proposal Network, RPN), and a head. Based on my previous experiments, I observed
that using Swin Transformer as the backbone yielded better performance. Therefore, Swin Transformer is
also adopted as the backbone in this assignment.

In the RPN, I adjusted parameters such as anchor sizes to observe their impact on detection performance.

In the head, I explored the integration of additional modules, such as the Convolutional Block Attention
Module (CBAM), in an effort to further enhance the model’s performance.

Additionally, I also incorporated Dice Loss into the mask branch.

## Setup and Preparation Process
### Step 1 : Environment Setup

#### 1. Use conda (Optimal)
```
conda env create -f environment.yml 
```

#### 2. Use pip
```
pip install -r requirements.txt
```

### Step 2 : Dataset Preparation
Place **hw3-data-release** folder at the same level as the **code** folder.

>project_root/ <br>
>├── code/ <br>
>├── hw3-data-release / <br>
>└── README.md

## Usage
### Hyperparameters
|Hyperparameter    | Value               |
|------------------|---------------------|
|model_name        |maskrcnn_swin_t_fpn|
| bs               | 2                   |
| epochs           | 50                  |
| lr               | 1e-4                |
| min_lr           | 1e-6                |
| weight_decay     | 0.005               |
| factor           | 0.1                 |

First, you need to navigate to the folder where our code is located in order to execute the preprocess, train and test commands.

```
cd HW3/code                      
```
### Preprocess

In this assignment, only a training set and a test set were provided, with no separate validation set.
Therefore, it was necessary to split the original training set into separate training and validation subsets.

```
python preprocess.py                 
```

It will generate **train.json** and **val.json** in **hw3-data-release** folder.

### Train

```
python train.py --data_path <Path to data> --train_json_path <Path to train JSON path> --val_json_path <Path to validation JSON path> --ckpt_path <Path to save checkpoint> --img_path <Path to save training curve image>                   
```
### Test

```
python test.py --test_data_path <Path to test data> --test_json_path <Path to test JSON path> --ckpt_path <Path to checkpoint> --result_path <Path to save the prediction result>              
```

## Performance snapshot
![Screenshot from 2025-04-16 16-30-16](https://github.com/user-attachments/assets/76c56b16-cf34-445e-98a5-19ba287b92f7)
