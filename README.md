# Perceptron Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation of the Perceptron algorithm - the simplest form of a neural network. This repository contains both theoretical explanations and practical implementations from scratch.

## 📖 About

The Perceptron is the fundamental building block of neural networks, introduced by Frank Rosenblatt in 1957. It's a binary classifier that can learn to separate linearly separable data points.

### What is a Perceptron?

A perceptron is a single-layer neural network that:
- Takes multiple inputs
- Applies weights and a bias
- Uses an activation function (typically step function)
- Outputs a binary classification (0 or 1)

## 🚀 Features

- **From Scratch Implementation**: Complete perceptron implementation without external ML libraries
- **Interactive Notebooks**: Step-by-step explanations with visualizations
- **Multiple Datasets**: Examples with different types of data
- **Educational Focus**: Detailed explanations of the learning process
- **Visualization**: Plots showing decision boundaries and learning progress

## 📁 Repository Contents

- `Perceptron.ipynb` - Main implementation with examples and visualizations
- `Perceptron_from_scratch.ipynb` - Detailed step-by-step implementation
- `README.md` - This documentation file

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abu-Taher01/Perceptron.git
   cd Perceptron
   ```

2. **Install required dependencies**:
   ```bash
   pip install numpy matplotlib pandas jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## 📚 Usage

### Basic Usage

```python
# Import the perceptron class
from perceptron import Perceptron

# Create a perceptron instance
perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)

# Train the perceptron
perceptron.fit(X_train, y_train)

# Make predictions
predictions = perceptron.predict(X_test)
```

### Example with Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train perceptron
perceptron = Perceptron()
perceptron.fit(X, y)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```

## 🧠 How It Works

### The Perceptron Algorithm

1. **Initialize**: Set weights and bias to small random values
2. **For each training example**:
   - Calculate output: `y_pred = step_function(w * x + b)`
   - Update weights: `w = w + learning_rate * (y_true - y_pred) * x`
   - Update bias: `b = b + learning_rate * (y_true - y_pred)`
3. **Repeat** until convergence or max iterations

### Mathematical Foundation

The perceptron learning rule:
```
w(t+1) = w(t) + α * (y - ŷ) * x
b(t+1) = b(t) + α * (y - ŷ)
```

Where:
- `w` = weights
- `b` = bias
- `α` = learning rate
- `y` = true label
- `ŷ` = predicted label
- `x` = input features

## 📊 Examples

The notebooks include examples with:
- **Linearly separable data**: Basic classification
- **XOR problem**: Demonstrates perceptron limitations
- **Real-world datasets**: Practical applications
- **Visualization**: Decision boundaries and learning curves

## 🔍 Key Concepts Covered

- **Linear Separability**: Understanding when perceptrons work
- **Learning Rate**: Impact on convergence
- **Decision Boundaries**: How perceptrons separate data
- **Limitations**: When perceptrons fail (XOR problem)
- **Gradient Descent**: Basic optimization principles

## 🎯 Learning Objectives

After working through this repository, you'll understand:
- How neural networks evolved from perceptrons
- The importance of activation functions
- Why multilayer networks are needed
- Basic machine learning concepts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abdullah Al Mamun**
- GitHub: [@Abu-Taher01](https://github.com/Abu-Taher01)
- LinkedIn: [Abdullah Al Mamun](https://www.linkedin.com/in/abdullah-al-mamun-003913205/)

---

⭐ **Star this repository if you found it helpful!** 
