#include <iostream>
#include <memory>
#include <cmath>
#include <vector>

// Define the Variable class
class Variable {
public:
    double value;            // The value of the variable
    double grad;             // The gradient of the variable (∂L/∂var)
    std::vector<std::pair<std::shared_ptr<Variable>, double> > parents; // Track dependencies for autograd

    // Constructor
    Variable(double value) : value(value), grad(0.0) {}

    // Add parent relationships to track gradients
    void add_parent(const std::shared_ptr<Variable>& parent, double weight) {
        parents.emplace_back(parent, weight);
    }

    // Backpropagation
    void backward(double chain_grad = 1.0) {
        grad += chain_grad; // Accumulate gradient from the chain rule
        for (const auto& parent_weight_pair : parents) {
            auto parent = parent_weight_pair.first;
            auto weight = parent_weight_pair.second;
            parent->backward(chain_grad * weight); // Propagate gradient to parents
        }
    }
};

// Overload basic operations to build the computational graph
std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto result = std::make_shared<Variable>(a->value * b->value);
    result->add_parent(a, b->value); // ∂(a*b)/∂a = b
    result->add_parent(b, a->value); // ∂(a*b)/∂b = a
    return result;
}

std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto result = std::make_shared<Variable>(a->value - b->value);
    result->add_parent(a, 1.0);  // ∂(a-b)/∂a = 1
    result->add_parent(b, -1.0); // ∂(a-b)/∂b = -1
    return result;
}

std::shared_ptr<Variable> pow(const std::shared_ptr<Variable>& a, double exp) {
    auto result = std::make_shared<Variable>(std::pow(a->value, exp));
    result->add_parent(a, exp * std::pow(a->value, exp - 1)); // ∂(a^exp)/∂a = exp * a^(exp-1)
    return result;
}

// Simple Neural Network Example with Autograd
void simple_neural_network_with_autograd(double learning_rate, int iterations) {
    // Initialize variables
    auto x = std::make_shared<Variable>(2.0);  // Input
    auto target = std::make_shared<Variable>(5.0); // Target
    auto w1 = std::make_shared<Variable>(0.5); // Weight

    for (int i = 0; i < iterations; ++i) {
        // Forward pass
        auto output = x * w1;                     // output = x * w1
        auto loss = pow(output - target, 2);      // loss = (output - target)^2

        // Backward pass
        w1->grad = 0;  // Reset gradients before backpropagation
        loss->backward(); // Perform backpropagation

        // Update weights using gradient descent
        w1->value -= learning_rate * w1->grad;

        // Print details for visualization
        std::cout << "Iteration " << i + 1 << ":\n";
        std::cout << "  Weight (w1): " << w1->value << "\n";
        std::cout << "  Gradient (∂L/∂w1): " << w1->grad << "\n";
        std::cout << "  Loss: " << loss->value << "\n\n";
    }
}

int main() {
    double learning_rate = 0.1; // Learning rate
    int iterations = 10;       // Number of iterations

    std::cout << "Simple Neural Network with Autograd...\n\n";
    simple_neural_network_with_autograd(learning_rate, iterations);

    return 0;
}