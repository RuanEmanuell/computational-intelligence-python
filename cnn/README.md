# Convolutional Neural Network (CNN)

# See the Portuguese version <a href="README-ptbr.md">here</a>

## What is a CNN?
A Convolutional Neural Network (CNN) is a neural network architecture specialized in image processing.

It learns visual patterns through convolution layers that detect edges, textures, and shapes.
For classification tasks, the network maps images to class probabilities.

## Implementation
This folder contains a Python implementation for image classification with **5 classes**.

The script supports two data modes:
- **Local dataset mode**: load images from `cnn/dataset/<class_name>/...`
- **Fallback mode**: use Fashion-MNIST filtered to 5 classes (`0` to `4`)

The implementation includes:
- Data loading and normalization
- CNN model definition with Keras/TensorFlow
- Training and validation split
- Test evaluation and confusion matrix
- Saving model and training curves

## Presentation
A presentation explaining the concept and implementation is also included:

<img width="1920" height="1080" alt="Inglês" src="https://github.com/user-attachments/assets/5ca99adb-369d-44dc-a60c-422f0f7cd4c6" />
<img width="1920" height="1080" alt="Inglês (3)" src="https://github.com/user-attachments/assets/21ea2186-9aec-4944-adce-b09713844a9f" />
<img width="1920" height="1080" alt="Inglês (2)" src="https://github.com/user-attachments/assets/0bf98bf5-49ae-4a2d-a771-9ef151544f1c" />
