# NYCU Computer Vision 2025 Spring HW4
StudentID: 313551093\
Name: 盧品樺

## Introduction

The objective of this assignment is image restoration, specifically targeting both rain- and snow-degraded
images using a single unified model. Building upon the **PromptIR** model, I improved its performance by
integrating the **Convolutional Block Attention Module (CBAM)** and modifying the number of blocks in
the architecture.

Additionally, I also incorporated the **perceptual loss** into the original loss function to
further enhance visual quality.

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
Place **hw4_realse_dataset** folder at the same level as the **code** folder.

>project_root/ <br>
>├── code/ <br>
>├── hw4_realse_dataset / <br>
>&nbsp; &nbsp; &nbsp; &nbsp; └── hw4_realse_dataset / <br>
>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ├── train  / <br>
>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; └──test  / <br>
>└── README.md

## Usage
### Hyperparameters
|Hyperparameter    | Value               |
|------------------|---------------------|
| bs               | 1                   |
| epochs           | 40                  |
| lr               | 5e-4                |
| min_lr           | 1e-6                |
| weight_decay     | 1e-3                |
| factor           | 0.2                 |

First, you need to navigate to the folder where our code is located in order to execute the preprocess, train and test commands.

```
cd HW4/code                      
```
### Preprocess

In this assignment, only a training set and a test set were provided, with no separate validation set.
Therefore, it was necessary to split the original training set into separate training and validation subsets.

```
python preprocess.py                 
```

It will generate **train.json** and **val.json** in **hw4_realse_dataset/hw4_realse_dataset** folder.

### Train

```
python train.py --data_path <Path to data> --train_json_path <Path to train JSON path> --val_json_path <Path to validation JSON path> --ckpt_path <Path to save checkpoint> --img_path <Path to save training curve image>                   
```
### Test

```
python test.py --test_data_path <Path to test data>  --ckpt_path <Path to checkpoint> --result_path <Path to save the prediction result>              
```

## Performance snapshot
![image](https://github.com/user-attachments/assets/6e47b499-5d71-49dd-9fce-174f8929723c)
