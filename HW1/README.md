# NYCU Computer Vision 2025 Spring HW1
StudentID: 313551093\
Name: 盧品樺

## Introduction
The goal of this assignment is to design and train a **model with ResNet as the backbone** for **image classification tasks**, where each image is assigned to one of 100 categories.
I selected ResNeXt101_32x8d as the backbone and improved its capabilities by incorporating dropout and the Convolutional Block Attention Module (CBAM). Additionally, I applied Mixup for data augmentation. Finally, the model achieved an accuracy of 0.91 on the validation set.

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
Place **data** folder at the same level as the **code** folder.

## Usage
### Hyperparameters

* bs : 32
* epochs : 50
* lr : 1e-4
* min_lr : 1e-6
* weight_decay : 0.045
* factor : 0.1
* mixup_alpha : 0.2

First, you need to navigate to the folder where our code is located in order to execute the train and test commands.

```
cd HW1/code                      
```

### Train

```
python train.py --train_path <Path to train data> --val_path <Path to validation data> --ckpt_path <Path to save checkpoint> --img_path <Path to save training curve image>                    
```

### Test

```
python test.py --test_path <Path to test data> --ckpt_path <Path to checkpoint> --csv_path <Path to save csv> --zip_path <Path to save zip>               
```

## Performance snapshot
![Screenshot from 2025-03-27 00-00-15](https://github.com/user-attachments/assets/94ea4a4e-92ad-43b0-b83a-84feebff8403)



