#include "cuda_runtime.h"
#include <string>
#include <cstdio>
#include <vector>
int main(int argc, char** argv){
 if(argc<2){std::fprintf(stderr,"usage: %s <path-to-msl-source>\n",argv[0]);return 64;}
 const std::string msl_path = argv[1];
 std::vector<int> mn(1,1000), mx(1,0), omn(32,-1), omx(32,-1);
 void *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr;
 cudaMalloc(&a,4);cudaMalloc(&b,4);cudaMalloc(&c,128);cudaMalloc(&d,128);
 cudaMemcpy(a,mn.data(),4,cudaMemcpyHostToDevice);cudaMemcpy(b,mx.data(),4,cudaMemcpyHostToDevice);
 static const cumetalKernelArgInfo_t kArgs[]={{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0}};
 const cumetalKernel_t kernel{msl_path.c_str(),"mm",4,kArgs};
 void *a1=a,*a2=b,*a3=c,*a4=d; void* args[]={&a1,&a2,&a3,&a4}; dim3 grid(1),block(32);
 if(cudaLaunchKernel(&kernel,grid,block,args,0,nullptr)!=cudaSuccess){printf("launch fail\n");return 1;}
 if(cudaDeviceSynchronize()!=cudaSuccess){printf("sync fail\n");return 1;}
 cudaMemcpy(mn.data(),a,4,cudaMemcpyDeviceToHost);cudaMemcpy(mx.data(),b,4,cudaMemcpyDeviceToHost);
 cudaMemcpy(omn.data(),c,128,cudaMemcpyDeviceToHost);cudaMemcpy(omx.data(),d,128,cudaMemcpyDeviceToHost);
 int bad=0;
 for(int i=0;i<32;++i){
   int expmn = i==0?1000:101-i;
   int expmx = i==0?0:i-1;
   if(omn[i]!=expmn||omx[i]!=expmx){if(bad<4)printf("lane %d omn=%d exp %d omx=%d exp %d\n",i,omn[i],expmn,omx[i],expmx);++bad;}
 }
 printf("final mn=%d (69) mx=%d (31) bad=%d %s\n",mn[0],mx[0],bad,(mn[0]==69&&mx[0]==31&&bad==0)?"OK":"BAD");
 return (mn[0]==69&&mx[0]==31&&bad==0)?0:1;
}
