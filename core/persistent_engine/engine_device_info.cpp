// engine_device_info.cpp
//
// Card 08B: Device info helper for benchmark reporting.

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace smoke {
namespace engine {

std::string engine_get_device_name_raw() {
    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice failed");
    }
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties failed");
    }
    return std::string(prop.name);
}

} // namespace engine
} // namespace smoke
