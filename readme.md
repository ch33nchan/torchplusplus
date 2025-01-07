# Ch33nChan Torch: Rebuilding PyTorch in C++

Welcome to Ch33nChan Torch, a project dedicated to recreating the magic of PyTorch from the ground up, purely in C++! This is a space where tensors, autograd, and neural networks are built step-by-step with maximum transparency and fun.

## 🌟 Vision

To replicate PyTorch-like functionality by:
1. Implementing tensor operations
2. Building an autograd engine
3. Creating neural network layers
4. Supporting custom loss functions, optimizers, and more

## 📌 Current State

We're at the starting line, implementing basic tensor operations, matrix multiplications, and a neural network capable of forward and backward passes.
This repository is for learning and experimentation—you'll understand how deep learning frameworks work internally.

## Directory Structure

```
└── ch33nchan-torch/
    ├── autograd.cpp
    ├── autograd_main.cpp
    ├── loss_func.cpp
    ├── mat_mul.cpp
    ├── ndaray-stride.cpp
    ├── ndarray.hpp
    └── neural_network.cpp
```

Each file serves a unique purpose in building this framework. Let's break it down with mathematical insights and internal workings.

## 1. ndarray.hpp: The Foundation of Tensors

### Purpose
Defines the core data structure for tensors, allowing for efficient storage and operations on multidimensional arrays. Tensors are the backbone of deep learning!

### Mathematical Explanation

A tensor is an n-dimensional array, generalized from:
- Scalar: Single value $$x \in \mathbb{R}$$
- Vector: 1D $$\vec{x} \in \mathbb{R}^n$$
- Matrix: 2D $$X \in \mathbb{R}^{m \times n}$$
- Higher dimensions: $$X \in \mathbb{R}^{d_1 \times d_2 \times ... \times d_n}$$

### Code Example

```cpp
class Tensor {
public:
    std::vector<float> values;  // Stores raw data
    std::vector<int> shape;     // Shape of the tensor (e.g., [2, 3] for a 2x3 matrix)
    std::vector<int> strides;   // Stride for efficient memory access

    Tensor(std::vector<int> shape) : shape(shape) {
        int total_elements = 1;
        for (int dim : shape) total_elements *= dim;
        values.resize(total_elements);
    }

    // Access an element
    float& at(int i, int j) {
        return values[i * strides[0] + j * strides[1]];
    }
};
```

### Internal Working
- Strides: Strides determine how we access elements in memory. For example:
  - Shape: $$(m, n)$$
  - Strides: $$(n, 1)$$
  - Index calculation: $$\text{index} = i \times \text{stride}[0] + j \times \text{stride}[1]$$

## 2. mat_mul.cpp: Matrix Multiplication

### Purpose
Implements matrix multiplication, a critical operation in deep learning.

### Mathematical Explanation

Given two matrices:
$$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$$

Their product is:
$$C = AB \in \mathbb{R}^{m \times p}$$

Where:
$$C_{ij} = \sum_{k=1}^n A_{ik}B_{kj}$$

### Code Example

```cpp
Tensor mat_mul(const Tensor& A, const Tensor& B) {
    int m = A.shape[0], n = A.shape[1], p = B.shape[1];
    Tensor C({m, p});

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A.values[i * A.strides[0] + k * A.strides[1]] *
                       B.values[k * B.strides[0] + j * B.strides[1]];
            }
            C.values[i * C.strides[0] + j * C.strides[1]] = sum;
        }
    }
    return C;
}
```

## 3. autograd.cpp & autograd_main.cpp: Automatic Differentiation

### Purpose
Implements autograd (automatic differentiation) to compute gradients for tensors.

### Mathematical Explanation

Using the chain rule, autograd computes derivatives:
$$\frac{\partial f}{\partial x} = \sum_{i} \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial x}$$

### Code Example

```cpp
void backward(Tensor& output) {
    output.grad[0] = 1.0f;  // Initialize gradient at output

    for (auto& op : output.history) {
        for (size_t i = 0; i < op.inputs.size(); ++i) {
            op.inputs[i]->grad[0] += op.gradient(op.inputs[i], output.grad[0]);
        }
    }
}
```

## 4. loss_func.cpp: Loss Functions

### Purpose
Defines loss functions for training models.

### Binary Cross-Entropy Loss

$$\mathcal{L}(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### Code Example

```cpp
float binary_cross_entropy(Tensor& predicted, Tensor& target) {
    float loss = 0.0f;
    for (int i = 0; i < predicted.values.size(); ++i) {
        float p = predicted.values[i];
        float t = target.values[i];
        loss -= t * log(p) + (1 - t) * log(1 - p);
    }
    return loss / predicted.values.size();
}
```

## 5. neural_network.cpp: Building a Simple Neural Network

### Purpose
Implements a fully connected neural network with forward and backward passes.

### Structure
- Linear layer: $$y = Wx + b$$
- Activation: Sigmoid $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Loss: Binary Cross-Entropy

### Code Example

```cpp
void forward_pass(Tensor& input, Tensor& weights, Tensor& bias, Tensor& output) {
    for (int i = 0; i < input.values.size(); ++i) {
        output.values[0] += input.values[i] * weights.values[i];
    }
    output.values[0] += bias.values[0];
    output.values[0] = 1 / (1 + exp(-output.values[0]));  // Sigmoid activation
}

// Training Loop
for (int epoch = 0; epoch < 100; ++epoch) {
    forward_pass(input, weights, bias, output);
    float loss = binary_cross_entropy(output, target);

    backward(output);  // Backpropagation
    update_parameters(weights, bias, learning_rate);

    std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
}
```

## How to Run

1. Compile the code:
```bash
g++ -std=c++11 -o neural_network neural_network.cpp
./neural_network
```

2. Enjoy the learning process!

## Roadmap
- Implement advanced tensor operations
- Add support for CNNs, RNNs, and optimizers
- Create a PyTorch-like API