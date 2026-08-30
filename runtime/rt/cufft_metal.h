#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <vector>

namespace cumetal::rt::fft_metal {

// cuFFT's advanced data layout, repeated here so the Metal path does not depend
// on cufft.cpp's internals. Element (x0, x1, x2) of batch b is at
// b*dist + stride * ((x0*embed[1] + x1)*embed[2] + x2); embed[0] is unused,
// matching cuFFT.
struct Layout {
    int embed[3] = {1, 1, 1};
    int stride = 1;
    long long dist = 0;
};

enum class Kind { kC2C, kR2C, kC2R };

struct Request {
    Kind kind = Kind::kC2C;
    int rank = 1;
    // Real dimensions for R2C/C2R, logical dimensions for C2C.
    int n[3] = {1, 1, 1};
    int batch = 1;
    Layout input;
    Layout output;
    bool inverse = false;  // C2C direction; ignored for R2C (forward) and C2R (inverse)
    cudaStream_t stream = nullptr;
    const void* idata = nullptr;
    void* odata = nullptr;
};

// Runs the transform on the Apple GPU.
//
// Returns false when this path declines -- unsupported precision, a policy
// setting, a pointer that does not resolve to a Metal buffer, a scratch
// allocation that failed. Declining is never a correctness statement: the caller
// runs its CPU implementation and gets the same answer. On false the output
// buffer is untouched.
bool execute(const Request& request);

// Unnormalized in-place radix-2 DFT over a power-of-two length, in double.
// Shared with the CPU Bluestein path so both fold the chirp the same way.
void fft_pow2(std::vector<double>& re, std::vector<double>& im, bool inverse);

}  // namespace cumetal::rt::fft_metal
