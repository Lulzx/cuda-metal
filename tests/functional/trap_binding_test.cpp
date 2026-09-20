#include "metal_backend.h"
#include "cumetal/common/kernel_abi.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    using namespace cumetal::metal_backend;
    std::string error;
    if (initialize(&error) != cudaSuccess) return 1;
    KernelArg collision;
    collision.binding_index = cumetal::abi::kTrapStatusBindingIndex;
    collision.bytes = {0, 0, 0, 0};
    LaunchConfig config{.grid = dim3(1,1,1), .block = dim3(1,1,1)};
    const auto status = launch_kernel(argv[1], "trap_probe", config, {collision}, nullptr, &error);
    if (status != cudaErrorInvalidValue || error.find("trap status binding") == std::string::npos) {
        std::cerr << "explicit binding collision was not rejected: " << status << " " << error << "\n";
        return 1;
    }
    GpuTimingResult timing;
    if (launch_kernel_timed(argv[1], "trap_probe", config, {}, &timing, &error) != cudaErrorNotSupported) {
        std::cerr << "unsupported timed trap path accepted\n";
        return 1;
    }
    return 0;
}
