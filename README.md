# Lightweight ResNet18 with Quantization on Caltech101

## Description

This project demonstrates how to apply dynamic quantization to a pre-trained ResNet18 model for efficient image classification on the Caltech101 dataset. Quantization reduces model size and inference time while maintaining accuracy.

## Dataset

[Caltech101](https://www.kaggle.com/datasets/varpit94/caltech-101) via `torchvision.datasets.Caltech101`—contains images from 101 object categories plus a background class.

## Installation

```bash
pip install torch torchvision

## Run
ResNet18 Accuracy (FP32): 0.5937
ResNet18 Accuracy (Quantized INT8): 0.5812
