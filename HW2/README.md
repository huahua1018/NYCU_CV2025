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

## Usage
### Hyperparameters

| Batch size       | 8                   |
| Epochs           | 18                  |
| Learning rate    | 1e-4                |
| Min learning rate| 1e-6                |
| Optimizer        | AdamW               |
| Weight decay     | 1e-3                |
| Scheduler        | ReduceLROnPlateau   |
| Factor           | 0.1                 |


First, you need to navigate to the folder where our code is located in order to execute the train and test commands.

```
cd HW2/code                      
```

### Train


### Test


## Performance snapshot
