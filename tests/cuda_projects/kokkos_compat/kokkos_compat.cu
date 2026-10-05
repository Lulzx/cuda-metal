#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

__constant__ int constant_value = 7;
__constant__ unsigned* constant_pointer;
struct alignas(16) LargeRecord {
    unsigned char padding[776];
    const float* values;
    unsigned count;
};
struct CounterRecord { unsigned* counter; unsigned long long unused; };
__global__ void captured_atomic(CounterRecord record) {
    unsigned old;
    asm volatile("atom.add.global.relaxed.gpu.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(record.counter), "r"(1u));
}
__global__ void captured_record(unsigned* output, LargeRecord record) {
    if (threadIdx.x == 0) {
        output[5] = record.padding[768];
        output[6] = record.padding[771];
        unsigned short word;
        __builtin_memcpy(&word, record.padding + 770, sizeof(word));
        output[7] = word;
    }
    alignas(16) unsigned char copied[sizeof(LargeRecord)];
    const auto* source = reinterpret_cast<const unsigned char*>(&record);
    for(unsigned i=0;i<sizeof(LargeRecord);++i) copied[i]=source[i];
    const auto* local = reinterpret_cast<const LargeRecord*>(copied);
    unsigned index=threadIdx.x;
    if(index<local->count) {
        float value=local->values[index];
        unsigned bits; __builtin_memcpy(&bits,&value,sizeof(bits));
        output[index]=bits;
    }
}
__global__ void ptx_shifts(unsigned long long* out, const unsigned* counts) {
    unsigned i = threadIdx.x, count = counts[i];
    unsigned a, b, c; unsigned long long d, e, f;
    asm("shl.b32 %0, %1, %2;" : "=r"(a) : "r"(0x80000001u), "r"(count));
    asm("shr.u32 %0, %1, %2;" : "=r"(b) : "r"(0x80000001u), "r"(count));
    asm("shr.s32 %0, %1, %2;" : "=r"(c) : "r"(0x80000001u), "r"(count));
    asm("shl.b64 %0, %1, %2;" : "=l"(d) : "l"(0x8000000000000001ull), "r"(count));
    asm("shr.u64 %0, %1, %2;" : "=l"(e) : "l"(0x8000000000000001ull), "r"(count));
    asm("shr.s64 %0, %1, %2;" : "=l"(f) : "l"(0x8000000000000001ull), "r"(count));
    out[i*6]=a; out[i*6+1]=b; out[i*6+2]=c;
    out[i*6+3]=d; out[i*6+4]=e; out[i*6+5]=f;
}
__device__ unsigned half_bits(__half value) {
    unsigned short bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}
__global__ void kokkos_compat(unsigned* out, const float* input) {
    __shared__ int shared_value;
    int local_value = 2;
    out[0] = __isGlobal(out);
    out[1] = __isShared(out);
    out[2] = __isShared(&shared_value);
    out[3] = __isGlobal(&shared_value);
    out[4] = __isLocal(&local_value);
    out[5] = __isGlobal(&local_value);
    out[6] = __isConstant(&constant_value);
    out[7] = __isShared(&constant_value);
    out[8] = half_bits(__float2half_rn(input[0]));
    out[9] = half_bits(__float2half_rn(input[1]));
    out[10] = static_cast<unsigned>(__half2int_rn(__float2half_rn(input[2])));
    out[11] = static_cast<unsigned>(__half2int_rz(__float2half_rn(input[2])));
    out[12] = static_cast<unsigned>(__bfloat162int_rz(__float2bfloat16(input[3])));
    out[13] = __float2bfloat16(input[4]).__x;
    out[14] = static_cast<unsigned>(std::ldexp(3u, 2));
    out[15] = static_cast<unsigned>(std::scalbn(3u, 2));
    atomicExch(constant_pointer + 16, 99u);
}
int main() {
    setenv("CUMETAL_USE_METAL_DEVICE_ADDRESSES", "1", 1);
    const float input[] = {0x1p-24f, 1.00048828125f, 3.5f, -3.75f, 1.00390625f};
    const unsigned expected[] = {1,0,1,0,1,0,1,0,1,0x3c00,4,3,static_cast<unsigned>(-3),0x3f80,12,12,99};
    unsigned *output = nullptr; float *device_input = nullptr;
    if (cudaMalloc(&output, sizeof(expected)) != cudaSuccess ||
        cudaMalloc(&device_input, sizeof(input)) != cudaSuccess ||
        cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(constant_pointer, &output, sizeof(output)) != cudaSuccess) return 1;
    kokkos_compat<<<1,1>>>(output, device_input);
    unsigned actual[17];
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    for (int i=0; i<17; ++i) if (actual[i] != expected[i]) {
        std::fprintf(stderr,"FAIL: compatibility result %d: %u != %u\n",i,actual[i],expected[i]); return 1;
    }
    LargeRecord record{}; record.values=device_input; record.count=5;
    record.padding[768]=0xa5; record.padding[770]=7; record.padding[771]=0x80;
    captured_record<<<1,32>>>(output,record);
    unsigned captured[8], input_bits[5];
    __builtin_memcpy(input_bits,input,sizeof(input_bits));
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
       cudaMemcpy(captured,output,sizeof(captured),cudaMemcpyDeviceToHost)!=cudaSuccess) return 1;
    for(unsigned i=0;i<5;++i) if(captured[i]!=input_bits[i]) {
        std::fprintf(stderr,"FAIL: aligned captured record index %u\n",i); return 1;
    }
    if(captured[5]!=0xa5 || captured[6]!=0x80 || captured[7]!=0x8007) {
        std::fprintf(stderr,"FAIL: captured parameter byte/halfword fields\n");return 1;
    }
    CounterRecord counter{output,0};
    if(cudaMemset(output,0,sizeof(unsigned))!=cudaSuccess) return 1;
    captured_atomic<<<1,32>>>(counter);
    unsigned atomic_result=0;
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
       cudaMemcpy(&atomic_result,output,sizeof(atomic_result),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       atomic_result!=32) { std::fprintf(stderr,"FAIL: captured zero-offset atomic\n"); return 1; }
    cudaFree(output); cudaFree(device_input);
    const unsigned counts[] = {0,1,31,32,33,63,64,65,~0u};
    unsigned* device_counts=nullptr; unsigned long long* shift_output=nullptr;
    if(cudaMalloc(&device_counts,sizeof(counts)) != cudaSuccess ||
       cudaMalloc(&shift_output,sizeof(counts)/sizeof(unsigned)*6*sizeof(unsigned long long)) != cudaSuccess ||
       cudaMemcpy(device_counts,counts,sizeof(counts),cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    ptx_shifts<<<1,9>>>(shift_output,device_counts);
    unsigned long long shifts[54];
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
       cudaMemcpy(shifts,shift_output,sizeof(shifts),cudaMemcpyDeviceToHost)!=cudaSuccess) return 1;
    for(unsigned i=0;i<9;++i) {
        unsigned n=counts[i];
        unsigned long long wanted[] = {
            n<32 ? unsigned(0x80000001u << n) : 0u,
            n<32 ? unsigned(0x80000001u >> n) : 0u,
            unsigned(int(0x80000001u) >> (n<32 ? n : 31)),
            n<64 ? 0x8000000000000001ull << n : 0ull,
            n<64 ? 0x8000000000000001ull >> n : 0ull,
            static_cast<unsigned long long>(static_cast<long long>(0x8000000000000001ull) >> (n<64 ? n : 63))};
        for(unsigned j=0;j<6;++j) if(shifts[i*6+j]!=wanted[j]) {
            std::fprintf(stderr,"FAIL: PTX shift count %u operation %u\n",n,j); return 1;
        }
    }
    cudaFree(device_counts); cudaFree(shift_output);
    std::puts("PASS: Kokkos compatibility on Apple GPU");
}
