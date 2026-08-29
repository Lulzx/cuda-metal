#include "metal_backend.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using cumetal::metal_backend::Buffer;
using cumetal::metal_backend::KernelArg;

KernelArg buffer_arg(const std::shared_ptr<Buffer> &buffer,
                     std::size_t binding = SIZE_MAX) {
  KernelArg argument;
  argument.kind = KernelArg::Kind::kBuffer;
  argument.buffer = buffer;
  argument.binding_index = binding;
  return argument;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2 || !std::filesystem::exists(argv[1])) {
    std::fprintf(stderr, "SKIP: usage: %s <constant-symbol.metallib>\n",
                 argv[0]);
    return 77;
  }
  using namespace cumetal::metal_backend;
  std::string error;
  if (initialize(&error) != cudaSuccess) {
    std::fprintf(stderr, "SKIP: Metal unavailable: %s\n", error.c_str());
    return 77;
  }

  std::shared_ptr<Buffer> output, constants, writable;
  if (allocate_buffer(sizeof(int), &output, &error) != cudaSuccess ||
      allocate_buffer(6976 * sizeof(int), &constants, &error) != cudaSuccess ||
      allocate_buffer(256 * sizeof(int), &writable, &error) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: typed symbol buffer allocation: %s\n",
                 error.c_str());
    return 1;
  }
  std::memset(constants->contents(), 0, constants->length());
  std::memset(writable->contents(), 0, writable->length());
  auto *constant_values = static_cast<int *>(constants->contents());
  auto *writable_values = static_cast<int *>(writable->contents());
  constant_values[0] = 11;
  constant_values[4096] = 31;
  writable_values[7] = 5;

  std::shared_ptr<Stream> stream;
  if (create_stream(&stream, &error) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: typed symbol stream creation: %s\n",
                 error.c_str());
    return 1;
  }
  const LaunchConfig config{
      .grid = dim3(1, 1, 1),
      .block = dim3(1, 1, 1),
      .shared_memory_bytes = 0,
      .provenance = "generic_nvvm_lowering",
      .semantic_quality = "exact",
  };
  const std::vector<KernelArg> constant_args{buffer_arg(output),
                                             buffer_arg(constants, 30)};
  if (launch_kernel(argv[1], "_Z17read_const_symbolPi", config, constant_args,
                    stream, &error) != cudaSuccess ||
      stream_synchronize(stream, &error) != cudaSuccess ||
      *static_cast<int *>(output->contents()) != 42) {
    std::fprintf(stderr, "FAIL: typed constant symbol got %d expected 42: %s\n",
                 *static_cast<int *>(output->contents()), error.c_str());
    return 1;
  }

  const std::vector<KernelArg> global_args{buffer_arg(output),
                                           buffer_arg(writable)};
  for (int launch = 0; launch < 2; ++launch) {
    if (launch_kernel(argv[1], "_Z23increment_device_symbolPi", config,
                      global_args, stream, &error) != cudaSuccess ||
        stream_synchronize(stream, &error) != cudaSuccess) {
      std::fprintf(stderr, "FAIL: typed writable symbol launch: %s\n",
                   error.c_str());
      return 1;
    }
  }
  const int output_value = *static_cast<int *>(output->contents());
  if (output_value != 11 || writable_values[7] != 11) {
    std::fprintf(
        stderr,
        "FAIL: typed writable symbol output=%d storage=%d expected 11\n",
        output_value, writable_values[7]);
    return 1;
  }
  if (destroy_stream(stream, &error) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: typed symbol stream destruction: %s\n",
                 error.c_str());
    return 1;
  }
  std::puts("PASS: typed direct constant and persistent writable CUDA symbols "
            "on Apple GPU");
  return 0;
}
