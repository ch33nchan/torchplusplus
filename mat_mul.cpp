#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept> // Include for std::invalid_argument

// Function to display a matrix
void displayMatrix(const std::vector< std::vector<int> >& matrix) {
    for (size_t i = 0; i < matrix.size(); ++i) {  // Use index-based loop
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << std::setw(4) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to perform matrix multiplication
std::vector< std::vector<int> > multiplyMatrices(const std::vector< std::vector<int> >& A, const std::vector< std::vector<int> >& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Initialize the result matrix with zeros
    std::vector< std::vector<int> > result(rowsA, std::vector<int>(colsB, 0));

    // Display the input matrices
    std::cout << "Matrix A:" << std::endl;
    displayMatrix(A);
    std::cout << "Matrix B:" << std::endl;
    displayMatrix(B);

    // Perform matrix multiplication and visualize each step
    std::cout << "\nMultiplying Matrices (Step by Step):\n";
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            // Calculate the dot product for element (i,j)
            int sum = 0;
            std::cout << "Step for result[" << i << "][" << j << "]: ";
            for (int k = 0; k < colsA; k++) {
                std::cout << A[i][k] << " * " << B[k][j];
                if (k < colsA - 1) {
                    std::cout << " + ";
                }
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
            std::cout << " = " << sum << std::endl;
        }
    }
    return result;
}

int main() {
    int n, m;
    std::cout << "Enter the number of rows and columns for Matrix A: ";
    std::cin >> n >> m;
    std::vector< std::vector<int> > A(n, std::vector<int>(m));

    std::cout << "Enter the elements of Matrix A (" << n << " x " << m << "):\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cin >> A[i][j];
        }
    }

    int p;
    std::cout << "Enter the number of columns for Matrix B (rows should be " << m << "): ";
    std::cin >> p;
    std::vector< std::vector<int> > B(m, std::vector<int>(p));

    std::cout << "Enter the elements of Matrix B (" << m << " x " << p << "):\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            std::cin >> B[i][j];
        }
    }

    try {
        // Perform the matrix multiplication and display the result
        std::vector< std::vector<int> > result = multiplyMatrices(A, B);

        // Display the result matrix
        std::cout << "\nResulting Matrix (A * B):" << std::endl;
        displayMatrix(result);
    } catch (const std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}