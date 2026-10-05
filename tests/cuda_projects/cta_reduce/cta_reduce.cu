#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__device__ __attribute__((noinline)) unsigned block_any(unsigned input) {
    unsigned result;
    asm volatile("{\n .reg .pred %%p;\n setp.ne.u32 %%p, %1, 0;\n bar.red.or.pred %%p, 0, %%p;\n selp.u32 %0, 1, 0, %%p;\n }"
                 : "=r"(result) : "r"(input));
    return result;
}
__device__ __attribute__((noinline)) unsigned block_count(unsigned input) {
    unsigned result;
    asm volatile("{\n .reg .pred %%p;\n setp.ne.u32 %%p, %1, 0;\n bar.red.popc.u32 %0, 0, %%p;\n }"
                 : "=r"(result) : "r"(input));
    return result;
}
__global__ void collective(unsigned* output) {
    unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned count = blockDim.x * blockDim.y * blockDim.z;
    unsigned result = 0;
    for (unsigned iteration = 0; iteration < 40; ++iteration) {
        result += block_any(tid == (iteration * 37 + blockIdx.x) % count);
        result += 100 * block_any(0);
        result += block_count(tid == (iteration * 37 + blockIdx.x) % count);
        result += 100 * block_count(0);
        result += block_count(1);
        result += block_count((tid & 1) == 0);
    }
    output[blockIdx.x * count + tid] = result;
}
__global__ void warp_prefix(unsigned* output) {
    __shared__ unsigned values[256], totals[8];
    unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned count = blockDim.x * blockDim.y * blockDim.z;
    unsigned lane = tid & 31, warp = tid >> 5, warps = count >> 5;
    for (unsigned repeat = 0; repeat < 12; ++repeat) {
        values[tid] = tid + 1;
        __syncwarp();
        for (unsigned stride = 1; stride < 32; stride *= 2) {
            unsigned previous = lane >= stride ? values[tid-stride] : 0;
            __syncwarp();
            values[tid] += previous;
            __syncwarp();
        }
        if (lane == 31) totals[warp] = values[tid];
        __syncthreads();
        if (tid < warps) {
            unsigned mask = (1u << warps) - 1;
            for (unsigned stride = 1; stride < warps; stride *= 2) {
                unsigned previous = tid >= stride ? totals[tid-stride] : 0;
                __syncwarp(mask);
                totals[tid] += previous;
                __syncwarp(mask);
            }
        }
        __syncthreads();
        output[blockIdx.x*count+tid] = values[tid] + (warp ? totals[warp-1] : 0);
        __syncthreads();
    }
}
__global__ void nested_exits(unsigned* output, unsigned count, unsigned outer_count) {
    unsigned acc=0, tid=threadIdx.x;
    for(unsigned outer=0;outer<outer_count;++outer) {
        __syncthreads();
        for(unsigned inner=0;inner<count;++inner) {
            if((tid%3)==0 && inner==2) goto tail;
            acc += inner + outer + 1;
        }
        acc += 100;
    tail:
        acc += 10;
        __syncthreads();
    }
    output[blockIdx.x*blockDim.x+tid]=acc;
}
int main() {
    for (dim3 block : {dim3(32), dim3(64), dim3(96), dim3(8,4,3), dim3(2,2,17)}) {
        unsigned count = block.x * block.y * block.z;
        unsigned *device = nullptr;
        if (cudaMalloc(&device, count * 4 * sizeof(unsigned)) != cudaSuccess) return 1;
        collective<<<4, block>>>(device);
        std::vector<unsigned> output(count * 4);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(output.data(), device, output.size()*sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        for (unsigned i=0; i<output.size(); ++i) if (output[i] != 40 * (2 + count + (count + 1)/2)) {
            std::fprintf(stderr,"FAIL: block (%u,%u,%u), result %u = %u has wrong CTA count\n", block.x,block.y,block.z,i,output[i]); return 1;
        }
        cudaFree(device);
    }
    for (dim3 block : {dim3(32),dim3(64),dim3(96),dim3(128),dim3(256),dim3(1,128),dim3(8,4,3)}) {
        unsigned count=block.x*block.y*block.z, *device=nullptr;
        if(cudaMalloc(&device,count*4*sizeof(unsigned))!=cudaSuccess) return 1;
        warp_prefix<<<4,block>>>(device);
        std::vector<unsigned> output(count*4);
        if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
           cudaMemcpy(output.data(),device,output.size()*sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess) return 1;
        for(unsigned i=0;i<output.size();++i) {
            unsigned tid=i%count, expected=(tid+1)*(tid+2)/2;
            if(output[i]!=expected) { std::fprintf(stderr,"FAIL: warp prefix count=%u index=%u actual=%u expected=%u\n",count,i,output[i],expected);return 1; }
        }
        cudaFree(device);
    }
    for(unsigned count : {1u,2u,3u,7u,11u}) {
        unsigned* device=nullptr;
        if(cudaMalloc(&device,3*128*sizeof(unsigned))!=cudaSuccess)return 1;
        nested_exits<<<3,128>>>(device,count,4);
        std::vector<unsigned> actual(3*128);
        if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
           cudaMemcpy(actual.data(),device,actual.size()*sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
        for(unsigned i=0;i<actual.size();++i) {
            unsigned expected=0;
            for(unsigned outer=0;outer<4;++outer) {
                unsigned limit=(i%128)%3==0 && count>2 ? 2 : count;
                for(unsigned inner=0;inner<limit;++inner)expected+=inner+outer+1;
                if(limit==count)expected+=100;
                expected+=10;
            }
            if(actual[i]!=expected) {std::fprintf(stderr,"FAIL: nested loop count=%u index=%u actual=%u expected=%u\n",count,i,actual[i],expected);return 1;}
        }
        cudaFree(device);
    }
    std::puts("PASS: CTA reductions across SIMD groups on Apple GPU");
}
