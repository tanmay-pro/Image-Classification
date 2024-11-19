# Image Classification Assignment

This repository contains the implementation of the Image Classification project as part of the **CS7.505: Computer Vision** course at the International Institute of Information Technology, Hyderabad (Spring 2024).

---

## Table of Contents
- [Objectives](#Objectives)
- [Requirements](#requirements)
- [Implementation](#implementation)
  - [Part 1: SIFT-BoVW-SVM](#part-1-sift-bovw-svm)
  - [Part 2: CNNs and Transformers](#part-2-cnns-and-transformers)

---

## Objectives

This project aims to implement and optimize image classification techniques, including the SIFT-BoVW-SVM approach and deep learning models like Convolutional Neural Networks (CNNs) and Transformers. The project focuses on implementing these methods and analyzing their performance on the MNIST handwritten digit recognition dataset.

---

## Requirements

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Weights & Biases (optional)

## Implementation

### Part 1: SIFT-BoVW-SVM

In this part, I have implemented the following:

1. **SIFT Feature Extraction**: Implement the SIFT detector and descriptor to extract features from the MNIST images.
2. **Bag-of-Visual-Words (BoVW)**: Perform BoVW to create a histogram representation of the images.
3. **Train SVM Model**: Train a linear SVM model for 10-way classification.
4. **Accuracy Analysis**: Analyze the classification accuracy as the number of clusters in the BoVW model is varied.
5. **Hyperparameter Experimentation**: Experiment with different SIFT and SVM hyperparameters, and observe their impact on classification accuracy.

### Part 2: CNNs and Transformers

In this part, I have implemented the following:

1. **CNN Training Setup**: Set up a modular codebase for training a CNN (LeNet) on the MNIST dataset.
2. **Performance Visualization**: Visualize the training and validation losses, as well as the test accuracy.
3. **Hyperparameter Tuning**: Experiment with different hyperparameter settings (batch size, learning rate, optimizer) and analyze their impact on the classification performance.
4. **Performance Comparison**: Compare the best-performing CNN model against the SIFT-BoVW-SVM approach.
5. **CNN Modification**: Modify the CNN model by adding more convolutional layers and observe the changes in performance.
6. **Transformer Evaluation**: Evaluate the classification accuracy of a 2-layer TransformerEncoder model on the MNIST dataset, using varying dataset sizes.
