#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <functional>

class NDArray {
private:
    std::unique_ptr<float[]> values;  // Dynamic array with automatic cleanup
    std::vector<int> shape;           // Dimensions sizes (e.g., [2,3] for 2x3 matrix)
    std::vector<int> strides;         // Steps to move in each dimension
    int ndim;                         // Number of dimensions
    int size;                         // Total elements

    void computeStrides() {
        strides.resize(ndim);
        int stride = 1;
        // Compute from last dimension to first
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = stride;          // Current dimension's stride
            stride *= shape[i];           // Multiply for next dimension
        }
    }

public:
    NDArray(const std::vector<int>& shape_) 
        : shape(shape_), 
          ndim(shape_.size()),
          size(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>())) {
        // Manually allocate memory instead of std::make_unique
        values.reset(new float[size]);
        computeStrides();  // Calculate strides
    }

    // Access element at given indices
    float& at(const std::vector<int>& indices) {
        // Safety checks
        if (indices.size() != ndim) {
            throw std::invalid_argument("Wrong number of indices");
        }

        int idx = 0;
        for (int i = 0; i < ndim; i++) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            idx += indices[i] * strides[i];  // Convert to linear index
        }
        return values[idx];
    }

    // Getter methods
    const std::vector<int>& getShape() const { return shape; }
    const std::vector<int>& getStrides() const { return strides; }
    int getDimensions() const { return ndim; }
    int getSize() const { return size; }

    // Fill array with a value
    void fill(float value) {
        for (int i = 0; i < size; i++) {
            values[i] = value;
        }
    }
};