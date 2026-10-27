#include "cuda_runtime.h"
#include <string>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
static unsigned short f2h(float f){unsigned u;std::memcpy(&u,&f,4);unsigned s=(u>>16)&0x8000;int e=int((u>>23)&0xff)-112;unsigned m=u&0x7fffff;if(e<=0)return (unsigned short)s;return (unsigned short)(s|(e<<10)|(m>>13));}
static float aval(int r,int k){return float((r*3+k)%5)*0.5f;}
static float bval(int k,int n){return float((k+n*2)%7)*0.25f;}
static float cval(int r,int n){return float((r+n)%3)*0.5f;}
int main(int argc, char** argv){
 if(argc<2){std::fprintf(stderr,"usage: %s <path-to-msl-source>\n",argv[0]);return 64;}
 const std::string msl_path = argv[1];
 std::vector<unsigned short> A(256),B(128); std::vector<float> C(128),out(128,-99.f);
 for(int r=0;r<16;++r)for(int k=0;k<16;++k)A[r*16+k]=f2h(aval(r,k));
 for(int n=0;n<8;++n)for(int k=0;k<16;++k)B[n*16+k]=f2h(bval(k,n));
 for(int r=0;r<16;++r)for(int n=0;n<8;++n)C[r*8+n]=cval(r,n);
 void *da=nullptr,*db=nullptr,*dc=nullptr,*dd=nullptr;
 cudaMalloc(&da,512);cudaMalloc(&db,256);cudaMalloc(&dc,512);cudaMalloc(&dd,512);
 cudaMemcpy(da,A.data(),512,cudaMemcpyHostToDevice);cudaMemcpy(db,B.data(),256,cudaMemcpyHostToDevice);cudaMemcpy(dc,C.data(),512,cudaMemcpyHostToDevice);cudaMemcpy(dd,out.data(),512,cudaMemcpyHostToDevice);
 static const cumetalKernelArgInfo_t kArgs[]={{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0}};
 const cumetalKernel_t kernel{msl_path.c_str(),"mm",4,kArgs};
 void *a1=da,*a2=db,*a3=dc,*a4=dd; void* args[]={&a1,&a2,&a3,&a4}; dim3 grid(1),block(32);
 if(cudaLaunchKernel(&kernel,grid,block,args,0,nullptr)!=cudaSuccess){printf("launch fail\n");return 1;}
 if(cudaDeviceSynchronize()!=cudaSuccess){printf("sync fail\n");return 1;}
 cudaMemcpy(out.data(),dd,512,cudaMemcpyDeviceToHost);
 float mx=0; for(int r=0;r<16;++r)for(int n=0;n<8;++n){float e=cval(r,n);for(int k=0;k<16;++k)e+=aval(r,k)*bval(k,n);float er=std::fabs(out[r*8+n]-e);if(er>mx)mx=er;}
 std::printf("PTX mma.sync m16n8k16 max_err=%g\n",mx);
 if(mx>0.01f){std::fprintf(stderr,"FAIL: PTX mma.sync result mismatch\n");return 1;}
 std::printf("PASS: PTX mma.sync m16n8k16\n"); return 0;
}
