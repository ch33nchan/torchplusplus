# torchplusplus 🚀

## Project Overview

The **end goal** of this project is to replicate PyTorch **from scratch**, using only **C++**. Currently, the repository focuses on building the **core foundations** required to handle tensors, implement activation functions, loss functions, and neural networks.

### Current Features:
1. **Tensor Creation**: Implementation of multi-dimensional arrays (tensors) with strides for efficient memory representation
2. **Activation Functions**: ReLU, Sigmoid, and LogSoftmax implementations
3. **Loss Functions**: Various loss functions including MSE, Cross Entropy, and Binary Cross Entropy
4. **Autograd Mechanism**: Automatic differentiation to compute gradients for optimization
5. **Simple Neural Networks**: Implementation of a feedforward neural network

---

## Mathematical Foundations

This project implements neural network components at their most fundamental level. Below is a detailed exploration of the mathematical foundations and implementations.

### Tensors

A **tensor** is a generalization of scalars, vectors, and matrices to higher dimensions:

- **0D Tensor**: Scalar (e.g., 3.0)
- **1D Tensor**: Vector (e.g., [1.0, 2.0, 3.0])
- **2D Tensor**: Matrix (e.g., [[1, 2], [3, 4]])
- **nD Tensor**: Multi-dimensional array

**Stride Mechanism:** Strides map multidimensional tensor indices to a flat array in memory.
- A tensor of shape `(2, 3)` with strides `[3, 1]` means the value at `(i, j)` maps to `i * 3 + j` in memory.

### Activation Functions

# ReLU (Rectified Linear Unit)

## Mathematical Definition

$$
\text{ReLU}(x) = \max(0, x)
$$

## Code Implementation

```cpp
float relu_forward(float x) {
    return std::max(0.0f, x);
}
```

## Example Usage

```cpp
float x = -5.0f;
float result = relu_forward(x);  // result = 0.0f

x = 3.0f;
result = relu_forward(x);  // result = 3.0f
```

Implementation:
```cpp
float relu_forward(float x) {
    return std::max(0.0f, x);
}

Tensor relu_forward(const Tensor& input) {
    Tensor output(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = relu_forward(input[i]);
    }
    return output;
}
```

#### Sigmoid
$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

Implementation:
```cpp
float sigmoid_forward(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

Tensor sigmoid_forward(const Tensor& input) {
    Tensor output(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = sigmoid_forward(input[i]);
    }
    return output;
}
```

#### LogSoftmax
Mathematical definition:

$$
\text{LogSoftmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)
$$

Implementation:
```cpp
Tensor log_softmax_forward(const Tensor& input) {
    Tensor output(input.shape());
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp sum with numerical stability
    float exp_sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        exp_sum += std::exp(input[i] - max_val);
    }
    float log_sum = std::log(exp_sum) + max_val;
    
    // Compute log_softmax
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] - log_sum;
    }
    return output;
}
```

### Loss Functions

#### Mean Squared Error (MSE)
Mathematical definition:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2
$$

Implementation:
```cpp
float mse_loss(const Tensor& y_pred, const Tensor& y_true) {
    float mse = 0.0f;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        float diff = y_pred[i] - y_true[i];
        mse += diff * diff;
    }
    return mse / y_pred.size();
}
```

#### Binary Cross Entropy
Mathematical definition:

$$
\text{BCE} = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
$$

Implementation:
```cpp
float binary_cross_entropy_loss(const Tensor& y_pred, const Tensor& y_true) {
    float loss = 0.0f;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        loss -= y_true[i] * std::log(y_pred[i] + 1e-7) + 
                (1 - y_true[i]) * std::log(1 - y_pred[i] + 1e-7);
    }
    return loss / y_pred.size();
}
```

#### Cross Entropy
Mathematical definition:

$$
\text{CE} = -\sum_{i=1}^C y_i \log(\hat{y}_i)
$$

where C is the number of classes.

Implementation:
```cpp
float cross_entropy_loss(const Tensor& y_pred, const Tensor& y_true) {
    float loss = 0.0f;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        loss -= y_true[i] * std::log(y_pred[i] + 1e-7);
    }
    return loss;
}
```

### Autograd

The autograd system computes gradients automatically using the chain rule:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

Implementation tracks computational history and computes gradients during backward pass.

### Neural Network Forward Pass

A typical neural network forward pass involves:

1. Input processing
2. Applying activation functions
3. Computing loss

Example implementation:
```cpp
class NeuralNetwork {
public:
    Tensor forward(const Tensor& input) {
        // First layer with ReLU activation
        Tensor hidden = relu_forward(input);
        
        // Output layer with sigmoid activation
        Tensor output = sigmoid_forward(hidden);
        
        return output;
    }
};
```

## File Structure

```
ch33nchan-torch/
├── autograd.cpp           # Implements automatic differentiation
├── autograd_main.cpp      # Entry point for testing
├── loss_functions.cpp     # Implements various loss functions
├── activation_funcs.cpp   # Implements activation functions
├── ndarray.hpp           # Header file for tensor operations
├── ndarray-stride.cpp    # Implements stride-based tensor indexing
└── neural_network.cpp    # Implements neural network components
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ch33nchan/torchplusplus.git
   ```

2. Compile and run:
   ```bash
   g++ -std=c++17 -o neural_network neural_network.cpp -O2
   ./neural_network
   ```

## Future Work

- Implement additional activation functions (e.g., Tanh, ELU)
- Add more loss functions (e.g., Huber Loss)
- Implement batch normalization
- Add dropout layers
- Optimize tensor operations

## Contributing

Feel free to fork and submit pull requests! Let's make this as powerful as PyTorch! 🚀
