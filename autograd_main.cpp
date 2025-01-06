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

    // Initialize the tensor with random values (for testing purposes)
    void initialize_values() {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // Random values between 0 and 1
        }
    }
};

// Example forward and backward passes using the `Tensor` class

void forward_pass(Tensor* out) {
    std::cout << "Forward Pass (ReLU Activation):\n";
    // Implement forward pass here (e.g., activation function, matrix multiplication, etc.)
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Input Value: " << out->values[i] << " -> ";
        out->values[i] = std::max(0.0f, out->values[i]);  // ReLU
        std::cout << "ReLU Output: " << out->values[i] << "\n";
    }
}

void backward_pass(Tensor* out) {
    std::cout << "\nBackward Pass (Gradient Computation - ReLU Derivative):\n";
    // Implement backward pass here (e.g., gradient computation)
    for (size_t i = 0; i < out->values.size(); ++i) {
        // Assuming that grad was set during the forward pass, we propagate gradients
        std::cout << "Gradient Input Value: " << out->grad[i] << " -> ";
        out->grad[i] = (out->values[i] > 0) ? out->grad[i] : 0;
        std::cout << "Gradient Output: " << out->grad[i] << "\n";
    }
}

// Sample gradient computation based on some operation, e.g., softmax or cross-entropy loss
void compute_gradients(Tensor* out) {
    std::cout << "\nComputing Gradients (Example Operation):\n";
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Value: " << out->values[i] << " -> ";
        out->grad[i] = out->values[i] * 0.5f;  // Example operation for gradient computation
        std::cout << "Gradient: " << out->grad[i] << "\n";
    }
}

// Example of a training loop where the gradients are updated based on operations
void train_step(Tensor* out) {
    forward_pass(out);  // Perform forward pass
    compute_gradients(out);  // Compute the gradients
    backward_pass(out);  // Backpropagate gradients
}

int main() {
    // Create an example tensor
    Tensor* out = new Tensor(10);  // Tensor of size 10

    // Initialize tensor values (optional)
    out->initialize_values();

    // Perform a single training step
    train_step(out);

    // Output the results
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