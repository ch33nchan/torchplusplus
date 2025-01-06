#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>  // For random number generation

// Utility function for generating random values between a given range
float rand_range(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

// Tensor class to represent a tensor and its gradient
class Tensor {
public:
    std::vector<float> values;
    std::vector<float> grad;
    std::vector<Tensor*> prevs;
    
    // Constructor to initialize values and gradients
    Tensor(size_t size) : values(size), grad(size, 0.0f) {}

    // Initialize the tensor with specific values for demonstration
    void initialize_values(const std::vector<float>& init_values) {
        values = init_values;
    }

    // Randomly initialize values for weights/biases
    void initialize_random(float min = -0.5, float max = 0.5) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = rand_range(min, max);
        }
    }
};

// ReLU activation function
void relu_forward(Tensor* out) {
    std::cout << "Applying ReLU activation:\n";
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Input: " << out->values[i] << " -> ";
        out->values[i] = std::max(0.0f, out->values[i]);  // ReLU: negative values become 0
        std::cout << "Output: " << out->values[i] << std::endl;
    }
}

// Sigmoid activation function
void sigmoid_forward(Tensor* out) {
    std::cout << "Applying Sigmoid activation:\n";
    for (size_t i = 0; i < out->values.size(); ++i) {
        std::cout << "Input: " << out->values[i] << " -> ";
        out->values[i] = 1.0f / (1.0f + expf(-out->values[i]));  // Sigmoid function
        std::cout << "Output: " << out->values[i] << std::endl;
    }
}

// Sigmoid derivative for backpropagation
void sigmoid_backward(Tensor* out) {
    std::cout << "Applying Sigmoid Backpropagation (Derivative):\n";
    for (size_t i = 0; i < out->values.size(); ++i) {
        out->grad[i] *= out->values[i] * (1.0f - out->values[i]);  // Derivative of sigmoid
        std::cout << "Gradient: " << out->grad[i] << std::endl;
    }
}

// Binary Cross-Entropy Loss function
float binary_cross_entropy_loss(const Tensor& output, const Tensor& target) {
    float loss = 0.0f;
    std::cout << "Calculating Binary Cross-Entropy Loss:\n";
    for (size_t i = 0; i < output.values.size(); ++i) {
        float bce = -target.values[i] * logf(output.values[i]) - (1.0f - target.values[i]) * logf(1.0f - output.values[i]);
        loss += bce;
        std::cout << "Loss for sample " << i << ": " << bce << std::endl;
    }
    return loss / output.values.size();
}

// Forward pass through a neural network (with weights and biases)
void forward_pass(Tensor& input, Tensor& weights, Tensor& bias, Tensor& output) {
    std::cout << "\nForward pass:\n";
    for (size_t i = 0; i < output.values.size(); ++i) {
        output.values[i] = 0.0f;
        for (size_t j = 0; j < input.values.size(); ++j) {
            output.values[i] += input.values[j] * weights.values[j];
        }
        output.values[i] += bias.values[i];
        std::cout << "Weighted sum for output " << i << ": " << output.values[i] << std::endl;
    }
}

// Training step
void train_step(Tensor& input, Tensor& target, Tensor& weights, Tensor& bias, Tensor& hidden, Tensor& output) {
    std::cout << "\nStarting Training Step:\n";
    
    // Forward Pass
    forward_pass(input, weights, bias, hidden);
    relu_forward(&hidden);
    forward_pass(hidden, weights, bias, output);
    sigmoid_forward(&output);
    
    // Calculate loss
    float loss = binary_cross_entropy_loss(output, target);
    std::cout << "Total Loss: " << loss << std::endl;

    // Backward Pass (gradient computation)
    sigmoid_backward(&output);

    // Backpropagate through the hidden layer
    hidden.grad.clear();
    hidden.grad.resize(hidden.values.size(), 0.0f);
    std::cout << "\nBackpropagating through hidden layer:\n";
    for (size_t i = 0; i < hidden.values.size(); ++i) {
        for (size_t j = 0; j < output.values.size(); ++j) {
            hidden.grad[i] += output.grad[j] * weights.values[i];  // Backprop through weights
        }
        hidden.grad[i] *= (hidden.values[i] > 0) ? 1 : 0;  // ReLU derivative
        std::cout << "Gradient for hidden node " << i << ": " << hidden.grad[i] << std::endl;
    }

    // Update weights and biases using gradient descent
    float learning_rate = 0.01f;
    std::cout << "\nUpdating Weights and Biases using Gradient Descent:\n";
    for (size_t i = 0; i < weights.values.size(); ++i) {
        weights.values[i] -= learning_rate * hidden.grad[i] * input.values[i];
        std::cout << "Updated weight " << i << ": " << weights.values[i] << std::endl;
    }

    for (size_t i = 0; i < bias.values.size(); ++i) {
        bias.values[i] -= learning_rate * output.grad[i];
        std::cout << "Updated bias " << i << ": " << bias.values[i] << std::endl;
    }
}

// Main function to create a simple neural network and perform training
int main() {
    srand(time(0));  // Seed random number generator for reproducibility

    // Define simple input and target data (XOR-like problem for demonstration)
    std::vector<float> input_data;
    input_data.push_back(0.0f);  // Example input for binary classification
    input_data.push_back(1.0f);
    
    std::vector<float> target_data;
    target_data.push_back(1.0f);  // Example target label (1 for positive class)

    Tensor input(2);  // Input Tensor with 2 features
    input.initialize_values(input_data);

    Tensor target(1);  // Target Tensor with 1 value
    target.initialize_values(target_data);

    // Initialize weights and biases randomly
    Tensor weights(2);  // Weights tensor for a simple 2-input, 1-output network
    weights.initialize_random();
    
    Tensor bias(1);  // Bias tensor for the output layer
    bias.initialize_random();

    Tensor hidden(1);  // Hidden layer with 1 node
    Tensor output(1);  // Output layer with 1 node

    // Perform one training step
    train_step(input, target, weights, bias, hidden, output);

    return 0;
}