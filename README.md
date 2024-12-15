
# GAN on MNIST with PyTorch

This project demonstrates the implementation of a Generative Adversarial Network (GAN) using PyTorch. The GAN is trained on the MNIST dataset to generate realistic-looking images of handwritten digits.

## Table of Contents
- [Introduction to GANs](#introduction-to-gans)
- [Implementation](#implementation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction to GANs
Generative Adversarial Networks (GANs) are a class of machine learning models where two neural networks, a generator and a discriminator, are trained simultaneously by competing with each other. 
- **Generator**: Tries to create realistic data.
- **Discriminator**: Tries to distinguish between real and generated data.

This adversarial training leads to a generator that can produce data indistinguishable from the real dataset.

### Theory
- The **generator** maps random noise to a data space resembling the training dataset (e.g., MNIST digits).
- The **discriminator** is a binary classifier trained to classify input as real or generated.

The GAN objective is to optimize:
```math
min_G max_D V(D, G) = E[log(D(x))] + E[log(1 - D(G(z)))]
```
where:
- \(D(x)\): Probability the discriminator assigns to real data.
- \(G(z)\): Generated data from random noise \(z\).

## Implementation
The notebook includes:
1. **Loading MNIST Dataset**: Prepares the dataset of handwritten digits.
2. **GAN Architecture**:
   - Generator: Fully connected layers with activation functions.
   - Discriminator: Fully connected layers with binary classification.
3. **Training Loop**: Alternates between training the discriminator and generator using the Binary Cross-Entropy loss.
4. **Results**: Generated images are visualized.

## How to Use
1. Clone this repository.
2. Install required dependencies: `torch`, `torchvision`, `matplotlib`.
3. Run the notebook `Gan_MNIST_Pytorch.ipynb` in Jupyter Notebook.

## Results
The trained GAN generates images resembling handwritten digits. Sample outputs are displayed in the notebook.

## Acknowledgments
This project uses the PyTorch framework and the MNIST dataset from Yann LeCun's database.
