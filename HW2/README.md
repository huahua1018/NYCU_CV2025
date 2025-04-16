# NYCU Computer Vision 2025 Spring HW2
StudentID: 313551093\
Name: 盧品樺

## Introduction

In this assignment, we need to implement a **digit recognition** model using the **Faster R-CNN** framework.

Faster R-CNN consists of three main components: a backbone, a neck (Region Proposal Network, RPN), and a head.

To enhance the model’s performance, I additionally experimented with Swin Transformer as the backbone.

In the RPN, I adjusted parameters such as anchor sizes to observe their effects on detection performance.

In the head, I explored adding extra layers to the box head and box predictor in an attempt to improve the model's performance.

Additionally, I also replaced the loss function with GIoU loss.

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
Place **nycu-hw2-data** folder at the same level as the **code** folder.

>project_root/ <br>
>├── code/ <br>
>├── nycu-hw2-data / <br>
>└── README.md

## Usage
### Hyperparameters
|Hyperparameter    | Value               |
|------------------|---------------------|
|model_name        |fasterrcnn_swin_t_fpn|
| bs               | 8                   |
| epochs           | 18                  |
| lr               | 1e-4                |
| min_lr           | 1e-6                |
| weight_decay     | 0.005               |
| factor           | 0.1                 |
|score_threshold   | 0.1                 |


First, you need to navigate to the folder where our code is located in order to execute the train and test commands.

```
cd HW2/code                      
```

### Train

```
python train.py --train_data_path <Path to train data> --train_json_path <Path to train JSON path> --val_data_path <Path to validation data> --val_json_path <Path to validation JSON path> --ckpt_path <Path to save checkpoint> --img_path <Path to save training curve image>                   
```
### Test

```
python test.py --test_data_path <Path to test data> --ckpt_path <Path to checkpoint> --result_path <Path to save the prediction result>              
```

## Performance snapshot
![Screenshot from 2025-04-16 16-30-16](https://github.com/user-attachments/assets/76c56b16-cf34-445e-98a5-19ba287b92f7)
