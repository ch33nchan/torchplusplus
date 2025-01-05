#include <iostream>
#include "ndarray.hpp"  // Save the previous code in this file

int main() {
    // Create a 2x3x2 array
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(2);
    
    NDArray arr(shape);
    
    // Fill with test value
    arr.fill(0.0f);
    
    // Set some values
    std::vector<int> idx1;
    idx1.push_back(0);
    idx1.push_back(0);
    idx1.push_back(0);
    arr.at(idx1) = 1.0f;
    
    std::vector<int> idx2;
    idx2.push_back(1);
    idx2.push_back(2);
    idx2.push_back(1);
    arr.at(idx2) = 42.0f;
    
    // Print information
    std::cout << "Array dimensions: " << arr.getDimensions() << std::endl;
    std::cout << "Total size: " << arr.getSize() << std::endl;
    
    // Print strides
    std::cout << "Strides: ";
    for (const auto& stride : arr.getStrides()) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    
    // Print values we set
    std::cout << "Value at (0,0,0): " << arr.at(idx1) << std::endl;
    std::cout << "Value at (1,2,1): " << arr.at(idx2) << std::endl;
    
    return 0;
}