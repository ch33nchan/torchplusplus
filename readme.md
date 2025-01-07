
# Ch33nchan-Torch 🚀

## Project Overview

The **end goal** of this project is to replicate PyTorch **from scratch**, using only **C++**. Currently, the repository focuses on building the **core foundations** required to handle tensors, perform mathematical operations, and implement neural networks.

### Current Features:
1. **Tensor Creation**: Implementation of multi-dimensional arrays (tensors) with strides for efficient memory representation.
2. **Matrix Multiplication**: Fundamental operations necessary for forward propagation and backpropagation.
3. **Autograd Mechanism**: Automatic differentiation to compute gradients for optimization.
4. **Loss Functions**: Implementation of Mean Squared Error (MSE).
5. **Simple Neural Networks**: Implementation of a feedforward neural network.

---

## Mathematical Foundations

This project delves into **tensor operations** and **neural network mechanics**, implementing them at the most fundamental level. Below is a deeper dive into the mathematical and technical details.

---

### Tensors

A **tensor** is a generalization of scalars, vectors, and matrices to higher dimensions. In this implementation:

- **0D Tensor**: Scalar (e.g., 3.0)
- **1D Tensor**: Vector (e.g., [1.0, 2.0, 3.0])
- **2D Tensor**: Matrix (e.g., [[1, 2], [3, 4]])
- **nD Tensor**: Multi-dimensional array

**Stride Mechanism:** Strides are used to map multidimensional tensor indices to a flat array in memory.

For example:
- A tensor of shape `(2, 3)` with strides `[3, 1]` means the value at `(i, j)` maps to `i * 3 + j` in memory.

### Code Example: Tensor Creation

```cpp
Tensor tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}); // A 2x3 tensor
tensor.print();  // Outputs the tensor in matrix format
```

---

### Matrix Multiplication

Matrix multiplication is the **core operation** of many machine learning algorithms, including neural networks.

#### Mathematical Definition:

Given two matrices \( A \) and \( B \):
\(
C_{ij} = \sum_{k=1}^n A_{ik} \cdot B_{kj}
\)

- **Dimensions**: If \( A \) is \( m 	imes n \) and \( B \) is \( n 	imes p \), then \( C \) will be \( m 	imes p \).

#### Code Example: Matrix Multiplication

Matrix multiplication is implemented in `mat_mul.cpp`. Here's the simplified logic:

```cpp
for (size_t i = 0; i < A_rows; ++i) {
    for (size_t j = 0; j < B_cols; ++j) {
        float sum = 0.0;
        for (size_t k = 0; k < A_cols; ++k) {
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
```

#### Example:

Input matrices:

\(
A = egin{bmatrix} 1 & 2 \ 3 & 4 \end{bmatrix}, \quad
B = egin{bmatrix} 5 & 6 \ 7 & 8 \end{bmatrix}
\)

Output:

\(
C = A \cdot B = egin{bmatrix} 19 & 22 \ 43 & 50 \end{bmatrix}
\)

---

### Autograd

Autograd is the mechanism for computing **gradients** automatically. This is crucial for backpropagation in neural networks.

#### Gradient Computation

If \( y = f(x) \), the gradient \( rac{\partial y}{\partial x} \) is computed using the **chain rule**:

\(
rac{\partial L}{\partial x} = rac{\partial L}{\partial y} \cdot rac{\partial y}{\partial x}
\)

#### Backward Pass

Gradients are propagated backward through the computation graph.

Example:
For \( z = x \cdot y \), where \( x = 2, y = 3 \):
\(
rac{\partial z}{\partial x} = y, \quad rac{\partial z}{\partial y} = x
\)

Code Snippet:

```cpp
float grad_x = y;  // Partial derivative w.r.t x
float grad_y = x;  // Partial derivative w.r.t y
```

---

### Loss Function

The **Mean Squared Error (MSE)** loss is defined as:

\(
	ext{MSE} = rac{1}{n} \sum_{i=1}^n (y_{	ext{pred}}^{(i)} - y_{	ext{true}}^{(i)})^2
\)

This is implemented in `loss_func.cpp`.

Code Example:

```cpp
float mse = 0.0;
for (size_t i = 0; i < y_pred.size(); ++i) {
    mse += std::pow(y_pred[i] - y_true[i], 2);
}
mse /= y_pred.size();
```

---

### Neural Network

The neural network is implemented as a feedforward network with one hidden layer.

#### Forward Propagation

1. Multiply inputs with weights.
2. Add biases.
3. Apply an activation function (ReLU).

Mathematical Representation:
\(
y = 	ext{ReLU}(X \cdot W + b)
\)

#### Backpropagation

Gradients are calculated using the chain rule. The weights and biases are updated using gradient descent:

\(
W_{	ext{new}} = W_{	ext{old}} - \eta \cdot rac{\partial L}{\partial W}
\)

where \( \eta \) is the learning rate.

Code Example:

```cpp
// Forward Pass
Tensor hidden = relu(matmul(input, weights_hidden) + bias_hidden);
Tensor output = matmul(hidden, weights_output) + bias_output;

// Backward Pass
Tensor grad_output = output - target;
weights_output -= learning_rate * matmul(hidden.T(), grad_output);
```

---

## File Structure

```
ch33nchan-torch/
├── autograd.cpp         # Implements automatic differentiation (gradients)
├── autograd_main.cpp    # Entry point to test autograd and tensors
├── loss_func.cpp        # Implements the Mean Squared Error (MSE) loss
├── mat_mul.cpp          # Implements matrix multiplication
├── ndarray.hpp          # Header file for tensor operations
├── ndaray-stride.cpp    # Implements stride-based tensor indexing
├── neural_network.cpp   # Implements a simple neural network
```

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ch33nchan-torch.git
   ```

2. Compile and run:
   ```bash
   g++ -std=c++17 -o neural_network neural_network.cpp -O2
   ./neural_network
   ```

3. Extend functionality by adding new operations to tensors or enhancing the autograd mechanism.

---

## Future Work

- Expand tensor operations (e.g., convolution, pooling).
- Implement more advanced loss functions.
- Add support for recurrent neural networks.
- Optimize tensor memory management.

---

## Contributing

Feel free to fork and submit pull requests! Let's make this as powerful as PyTorch! 🚀
