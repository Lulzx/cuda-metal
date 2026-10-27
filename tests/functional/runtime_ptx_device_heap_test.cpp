#include "cuda_runtime.h"
#include <string>
#include <cstdio>
#include <cstdint>
#include <vector>
// Exercises the typed PTX device-heap channel: the kernel takes the
// hidden __cumetal_device_heap buffer as its last argument. Registered
// launches append that buffer automatically; this direct-launch gate
// supplies an explicitly initialized heap (bump=16, free head=0,
// capacity=size) so the compiler lowering and allocator are tested
// without the offline Metal toolchain.
int main(int argc, char** argv){
 if(argc<2){std::fprintf(stderr,"usage: %s <path-to-msl-source>\n",argv[0]);return 64;}
 const std::string msl_path = argv[1];
 std::vector<int> out(3,-99); std::vector<unsigned long long> addrs(3,0);
 void *dout=nullptr,*daddrs=nullptr,*dheap=nullptr;
 const uint32_t heap_size=1024*1024;
 cudaMalloc(&dout,12); cudaMalloc(&daddrs,24); cudaMalloc(&dheap,heap_size);
 cudaMemcpy(dout,out.data(),12,cudaMemcpyHostToDevice);
 cudaMemset(dheap,0,heap_size);
 uint32_t hdr[3]={16,0,heap_size};
 cudaMemcpy(dheap,hdr,12,cudaMemcpyHostToDevice);
 static const cumetalKernelArgInfo_t kArgs[]={{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BYTES,4},{CUMETAL_ARG_BUFFER,0}};
 const cumetalKernel_t kernel{msl_path.c_str(),"heapk",4,kArgs};
 void *a1=dout,*a2=daddrs; unsigned bb=2*1024*1024; void *a4=dheap; void* args[]={&a1,&a2,&bb,&a4}; dim3 grid(1),block(32);
 if(cudaLaunchKernel(&kernel,grid,block,args,0,nullptr)!=cudaSuccess){printf("launch fail\n");return 1;}
 if(cudaDeviceSynchronize()!=cudaSuccess){printf("sync fail\n");return 1;}
 cudaMemcpy(out.data(),dout,12,cudaMemcpyDeviceToHost);
 cudaMemcpy(addrs.data(),daddrs,24,cudaMemcpyDeviceToHost);
 int reuse = (addrs[0]!=0 && addrs[0]==addrs[1]) ? 1 : 0;
 printf("sum=%d (84) sum2=%d (828) reuse=%d (1) bignull=%d (9) %s\n",out[0],out[1],reuse,out[2],
   (out[0]==84&&out[1]==828&&reuse==1&&out[2]==9)?"OK":"BAD");
 return (out[0]==84&&out[1]==828&&reuse==1&&out[2]==9)?0:1;
}
