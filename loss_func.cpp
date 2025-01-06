#include <iostream>
#include <vector>
#include <cmath>  // For pow function, abs, and log

// Mean Squared Error (MSE) loss
// Measures the average squared difference between predicted and actual values.
// Smaller MSE indicates better predictions.
float meanSquaredError(const std::vector<float>& predicted, const std::vector<float>& actual) {
    // Check if sizes of predicted and actual match
    if (predicted.size() != actual.size()) {
        std::cerr << "Error: Predicted and Actual vectors must have the same size!" << std::endl;
        return -1;  // Return -1 to indicate an error
    }
    float loss = 0.0f;
    // Calculate the squared difference for each element
    for (size_t i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - actual[i];
        loss += diff * diff;  // Add squared difference
    }
    return loss / predicted.size();  // Average the squared differences
}

// Mean Absolute Error (MAE) loss
// Measures the average absolute difference between predicted and actual values.
// More robust to outliers than MSE.
float meanAbsoluteError(const std::vector<float>& predicted, const std::vector<float>& actual) {
    if (predicted.size() != actual.size()) {
        std::cerr << "Error: Predicted and Actual vectors must have the same size!" << std::endl;
        return -1;  // Return -1 to indicate an error
    }
    float loss = 0.0f;
    // Calculate the absolute difference for each element
    for (size_t i = 0; i < predicted.size(); i++) {
        loss += std::abs(predicted[i] - actual[i]);  // Add absolute difference
    }
    return loss / predicted.size();  // Average the absolute differences
}

// Cross-Entropy Loss (For binary classification)
// Measures how well predicted probabilities match the true labels (0 or 1).
// Used in classification tasks with probabilistic outputs.
float crossEntropyLoss(const std::vector<float>& predicted, const std::vector<float>& actual) {
    if (predicted.size() != actual.size()) {
        std::cerr << "Error: Predicted and Actual vectors must have the same size!" << std::endl;
        return -1;  // Return -1 to indicate an error
    }
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); i++) {
        // Clip predicted values to avoid issues with log(0) (use small values like 1e-7)
        float pred = std::max(1e-7f, std::min(1 - 1e-7f, predicted[i]));
        // Calculate cross-entropy for this instance
        loss += -actual[i] * std::log(pred) - (1 - actual[i]) * std::log(1 - pred);
    }
    return loss / predicted.size();  // Average the cross-entropy loss
}

// Hinge Loss (For binary classification)
// Used in Support Vector Machines (SVMs). Penalizes predictions that are on the wrong side
// of the decision boundary by computing the margin.
float hingeLoss(const std::vector<float>& predicted, const std::vector<float>& actual) {
    if (predicted.size() != actual.size()) {
        std::cerr << "Error: Predicted and Actual vectors must have the same size!" << std::endl;
        return -1;  // Return -1 to indicate an error
    }
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); i++) {
        // Calculate the margin: 1 - y_actual * y_predicted
        float margin = 1.0f - actual[i] * predicted[i];  // y_actual should be -1 or 1
        if (margin > 0) {  // Only add to loss if margin is greater than 0
            loss += margin;
        }
    }
    return loss / predicted.size();  // Average the hinge loss
}

int main() {
    // Example data for regression tasks
    // `predicted` contains the model's predicted values
    // `actual` contains the true values
    std::vector<float> predicted = {3.0, 2.5, 4.0, 5.1};
    std::vector<float> actual = {3.1, 2.7, 4.2, 5.0};

    // Example data for classification tasks
    // `predicted_class` contains probabilities output by the model (between 0 and 1)
    // `actual_class` contains ground-truth binary labels (0 or 1)
    std::vector<float> predicted_class = {0.8, 0.2, 0.7, 0.9};
    std::vector<float> actual_class = {1, 0, 1, 1};

    // Calculate Mean Squared Error (MSE) loss
    float mse_loss = meanSquaredError(predicted, actual);
    std::cout << "Mean Squared Error Loss: " << mse_loss << std::endl;

    // Calculate Mean Absolute Error (MAE) loss
    float mae_loss = meanAbsoluteError(predicted, actual);
    std::cout << "Mean Absolute Error Loss: " << mae_loss << std::endl;

    // Calculate Cross-Entropy Loss (for binary classification)
    float ce_loss = crossEntropyLoss(predicted_class, actual_class);
    std::cout << "Cross-Entropy Loss: " << ce_loss << std::endl;

    // Calculate Hinge Loss (for binary classification)
    float hinge_loss = hingeLoss(predicted_class, actual_class);
    std::cout << "Hinge Loss: " << hinge_loss << std::endl;

    return 0;  // End of program
}