# Madaline

# See the Portuguese version <a href="README-ptbr.md">here</a>

## What is a Madaline?
The **Madaline (Multiple ADAptive LInear NEuron)** is a type of artificial neural network used for pattern recognition and multi-class classification.

It is an extension of the Perceptron, using multiple neurons to classify more than two classes.

It learns by adjusting its weights based on the error between the predicted output and the expected output:
- If the prediction is correct, no changes are made
- If the prediction is wrong, the weights are updated to reduce the error

## Implementation
This project implements a Madaline neural network to recognize letters from **A to Z**.

The implementation includes:
- Pattern recognition using **7x7 matrices**
- Conversion of pixels into numerical values (`1` and `-1`)
- Multi-class classification (26 letters)
- Training using error correction

## How does it work?
- Each letter is represented as a **7x7 matrix (49 inputs)**
- Values are converted to:
  - `1` → active pixel  
  - `-1` → inactive pixel  
- The network has:
  - **49 inputs**
  - **26 outputs (one per letter)**

During training:
- The correct letter receives **+1**
- The others receive **-1**
- Weights are adjusted only when errors occur

After training, the network predicts the letter based on the highest activation value.

## Presentation
A presentation explaining the concept and implementation is also included:

<img width="1920" height="1080" alt="Inglês"     src="https://github.com/user-attachments/assets/e84b1698-2b62-4d17-b23c-326fff212694" />
<img width="1920" height="1080" alt="Inglês (2)" src="https://github.com/user-attachments/assets/4c6d4c76-40bb-415f-8d3d-5802ed926877" />
<img width="1920" height="1080" alt="Inglês (3)" src="https://github.com/user-attachments/assets/0666e339-f835-487b-9f93-d17d560056a3" />
<img width="1920" height="1080" alt="Inglês (4)" src="https://github.com/user-attachments/assets/ce3eb417-3f35-4ce9-b9ca-06e486d81151" />
<img width="1920" height="1080" alt="Inglês (5)" src="https://github.com/user-attachments/assets/5b7568a6-c8d8-49ac-9747-15724b53e892" />