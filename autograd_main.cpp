#include <iostream>
#include <vector>
#include <cmath>  // For expf

// Tensor class to represent a tensor and its gradient
class Tensor {
public:
    std::vector<float> values;  // Stores the actual data (e.g., values of a neural network layer)
    std::vector<float> grad;    // Stores the gradients (derivatives w.r.t. some loss function)
    std::vector<Tensor*> prevs; // List of previous tensors (for backpropagation)

    // Constructor to initialize values and gradients
    Tensor(size_t size) : values(size), grad(size, 0.0f) {}  // Initialize data and set gradients to 0

    // Initialize the tensor with specific values for demonstration
    void initialize_values(const std::vector<float>& init_values) {
        values = init_values;
    }
};

// Simple ReLU activation function
void forward_pass(Tensor* out) {
    std::cout << "Forward Pass (ReLU Activation):\n";
    // Apply ReLU activation to the tensor
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Input Value: " << out->values[i] << " -> ";
        out->values[i] = std::max(0.0f, out->values[i]);  // ReLU: negative values become 0
        std::cout << "ReLU Output: " << out->values[i] << "\n";
    }
}

// Simple gradient computation (example: scaling by a constant)
void compute_gradients(Tensor* out) {
    std::cout << "\nComputing Gradients (Example Operation):\n";
    // Here, we will just scale the values as a simple example
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Value: " << out->values[i] << " -> ";
        out->grad[i] = out->values[i] * 0.5f;  // Simple gradient operation for demo
        std::cout << "Gradient: " << out->grad[i] << "\n";
    }
}

// Backpropagation to propagate gradients (using ReLU derivative)
void backward_pass(Tensor* out) {
    std::cout << "\nBackward Pass (ReLU Gradient Calculation):\n";
    // Applying ReLU gradient rule (1 if positive, 0 if negative)
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Gradient of ReLU Input Value: " << out->grad[i] << " -> ";
        out->grad[i] = (out->values[i] > 0) ? out->grad[i] : 0;  // ReLU derivative
        std::cout << "Gradient after ReLU Derivative: " << out->grad[i] << "\n";
    }
}

// A training step simulating forward pass, gradient computation, and backward pass
void train_step(Tensor* out) {
    forward_pass(out);  // Perform forward pass
    compute_gradients(out);  // Compute gradients (simple example)
    backward_pass(out);  // Backpropagate gradients using ReLU
}

int main() {
    // Step 1: Create a tensor with simple input values
    std::vector<float> init_values = {0.5, -0.2, 0.8, -0.5, 1.0};  // Simple initial values for the tensor
    Tensor* out = new Tensor(init_values.size());  // Initialize tensor with size matching input

    // Step 2: Initialize tensor with values
    out->initialize_values(init_values);

    // Step 3: Perform one training step (forward pass, compute gradients, backward pass)
    train_step(out);

    // Step 4: Output final values and gradients
    std::cout << "\nFinal Values and Gradients after One Training Step:\n";
    std::cout << "Values: ";
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << out->values[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradients: ";
    for (size_t i = 0; i < out->grad.size(); ++i) {
        std::cout << out->grad[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete out;

    return 0;
}