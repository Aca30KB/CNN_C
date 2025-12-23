# CNN_C

A simple implementation of a neural network in pure C for classifying the MNIST dataset.  
This project is educational in nature and demonstrates how to build a complete training and testing pipeline without relying on external machine learning libraries.

## ✨ Features
- Implementation of a **Multi-Layer Perceptron (MLP)** (784 → 128 → 10) with one hidden layer
- Activation functions: **ReLU** in the hidden layer, **Softmax** at the output
- Loss function: **Cross-Entropy**
- Optimization: **Stochastic Gradient Descent (SGD)** with mini-batch support
- Weight initialization: **He initialization**
- Data loading from CSV files (`mnist_train.csv`, `mnist_test.csv`)
- Manual memory management in C

## 📂 Project Structure
- `main.c` – entry point, training and testing loop
- `cnn.c/h` – neural network implementation (forward pass, backpropagation)
- `data.c/h` – CSV data loading
- `image.c/h` – image structure and memory allocation

## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/Aca30KB/CNN_C.git
   cd CNN_C
