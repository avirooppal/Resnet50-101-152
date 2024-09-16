# ResNet Implementation in PyTorch

This repository contains an implementation of the Residual Network (ResNet) in PyTorch. ResNet is a popular deep learning architecture used for image classification, which introduces residual connections (or skip connections) to help improve the training of deep networks.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Model Architectures](#model-architectures)
- [Usage](#usage)
- [Testing](#testing)
- [References](#references)

## Overview
This implementation supports ResNet50, ResNet101, and ResNet152 architectures. Each of these models can be initialized with custom input image channels and the number of output classes.

The core of the implementation is based on the original ResNet paper: 
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

## Requirements

- Python 3.x
- PyTorch >= 1.7.0
- TorchVision (optional, for dataset handling)

Install the required Python packages using `pip`:

```bash
pip install torch torchvision
```

## Model Architectures

The following ResNet variants are supported:

- **ResNet50**: Uses 50 layers.
- **ResNet101**: Uses 101 layers.
- **ResNet152**: Uses 152 layers.

### Example Usage:

To create a model with 3 input image channels and 1000 output classes (default configuration):

```python
from resnet import Resnet50, Resnet101, Resnet152

model = Resnet50(img_channels=3, num_classes=1000)
```

## Usage

### 1. Initialize the model

```python
model = Resnet50(img_channels=3, num_classes=1000)  # For ResNet50
```

### 2. Forward Pass

```python
import torch

# Random input tensor of size [batch_size, channels, height, width]
x = torch.randn(2, 3, 224, 224)

# Perform forward pass
output = model(x)

# Output size will be [batch_size, num_classes]
print(output.shape)
```

### 3. Training and Evaluation

You can train this model by using a standard training loop, applying loss functions like `CrossEntropyLoss` and optimizers like `Adam` or `SGD` from PyTorch.

Example:
```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Assuming you have a DataLoader
for inputs, labels in train_loader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## Testing

A simple test function is included to test the forward pass of ResNet50:

```bash
python resnet.py
```

The output should print the shape of the result from the model, ensuring everything is set up correctly.

```python
torch.Size([2, 1000])
```

## References
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
```
