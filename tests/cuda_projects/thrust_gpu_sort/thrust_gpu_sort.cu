#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <string>
#include <vector>

struct Descending {
    __host__ __device__ bool operator()(int a,int b) const { return a>b; }
};
struct Iterator {
    using value_type=int; using difference_type=ptrdiff_t; using pointer=int*;
    using reference=int&; using iterator_category=std::random_access_iterator_tag;
    int* data; ptrdiff_t offset;
    __host__ __device__ int& operator[](size_t i) const {return data[offset+i];}
    __host__ __device__ ptrdiff_t operator-(Iterator other) const {return offset-other.offset;}
};
__global__ void producer(int* data,unsigned count) {
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        data[i]=(i==0?INT_MIN:i==1?INT_MAX:int((i*7919)%257)-128);
}
int main() {
    cudaStream_t stream=nullptr; int* device=nullptr;
    if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess || cudaMalloc(&device,4099*sizeof(int))!=cudaSuccess) return 1;
    for(unsigned count : {0u,1u,17u,257u,4097u}) {
        producer<<<9,128,0,stream>>>(device,count);
        std::vector<int> expected(count),actual(count);
        for(unsigned i=0;i<count;++i)expected[i]=(i==0?INT_MIN:i==1?INT_MAX:int((i*7919)%257)-128);
        thrust::sort(thrust::cuda::par.on(stream),Iterator{device,0},Iterator{device,count});
        if(count && cudaMemcpy(actual.data(),device,count*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
        std::sort(expected.begin(),expected.end());
        if(actual!=expected) {std::fprintf(stderr,"FAIL: ascending GPU sort count=%u\n",count);return 1;}
        thrust::sort(thrust::cuda::par.on(stream),thrust::device_pointer_cast(device),thrust::device_pointer_cast(device+count),Descending{});
        if(count && cudaMemcpy(actual.data(),device,count*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
        std::sort(expected.begin(),expected.end(),Descending{});
        if(actual!=expected) {std::fprintf(stderr,"FAIL: descending GPU sort count=%u\n",count);return 1;}
    }
    bool reversed=false,invalid_stream=false,unsupported=false;
    try {thrust::sort(thrust::cuda::par.on(stream),device+5,device);}catch(const std::runtime_error& e){reversed=std::string(e.what()).find("cudaErrorInvalidValue")!=std::string::npos;}
    try {thrust::sort(thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(0xdeadbeef)),device,device);}catch(const std::runtime_error&){invalid_stream=true;}
    cudaGetLastError();
    try {thrust::sort_by_key(thrust::cuda::par.on(stream),device,device+5,device+5);}catch(const std::runtime_error& e){unsupported=std::string(e.what()).find("cudaErrorNotSupported")!=std::string::npos;}
    if(!reversed||!invalid_stream||!unsupported){std::fprintf(stderr,"FAIL: GPU sort negative paths reversed=%d stream=%d unsupported=%d\n",reversed,invalid_stream,unsupported);return 1;}
    if(cudaFree(device)!=cudaSuccess||cudaStreamDestroy(stream)!=cudaSuccess)return 1;
    std::puts("PASS: CUDA-policy Thrust sort on Apple GPU");
}
