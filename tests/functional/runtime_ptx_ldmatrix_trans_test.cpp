#include "cuda_runtime.h"
#include <string>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
static unsigned short f2h(float f){unsigned u;std::memcpy(&u,&f,4);unsigned s=(u>>16)&0x8000;int e=int((u>>23)&0xff)-112;unsigned m=u&0x7fffff;if(e<=0)return (unsigned short)s;return (unsigned short)(s|(e<<10)|(m>>13));}
static float aval(int r,int c){return float((r*5+c)%9)*0.25f;}
int main(int argc, char** argv){
 if(argc<2){std::fprintf(stderr,"usage: %s <path-to-msl-source>\n",argv[0]);return 64;}
 const std::string msl_path = argv[1];
 std::vector<unsigned short> A(256); std::vector<float> out(256,-99.f);
 for(int r=0;r<16;++r)for(int c=0;c<16;++c)A[r*16+c]=f2h(aval(r,c));
 void *da=nullptr,*dd=nullptr;
 cudaMalloc(&da,512);cudaMalloc(&dd,1024);
 cudaMemcpy(da,A.data(),512,cudaMemcpyHostToDevice);cudaMemcpy(dd,out.data(),1024,cudaMemcpyHostToDevice);
 static const cumetalKernelArgInfo_t kArgs[]={{CUMETAL_ARG_BUFFER,0},{CUMETAL_ARG_BUFFER,0}};
 const cumetalKernel_t kernel{msl_path.c_str(),"ldm",2,kArgs};
 void *a1=da,*a2=dd; void* args[]={&a1,&a2}; dim3 grid(1),block(32);
 if(cudaLaunchKernel(&kernel,grid,block,args,0,nullptr)!=cudaSuccess){printf("launch fail\n");return 1;}
 if(cudaDeviceSynchronize()!=cudaSuccess){printf("sync fail\n");return 1;}
 cudaMemcpy(out.data(),dd,1024,cudaMemcpyDeviceToHost);
 float mx=0; int bad=-1;
 for(int i=0;i<256;++i){int r=i/16,c=i%16; float expv=aval((r/8)*8+(c%8),(c/8)*8+(r%8)); float er=std::fabs(out[i]-expv);if(er>mx){mx=er;bad=i;}}
 std::printf("PTX ldmatrix m8n8.x4.trans max_err=%g\n",mx);
 if(mx>0.01f){std::fprintf(stderr,"FAIL: PTX ldmatrix result mismatch\n");return 1;}
 std::printf("PASS: PTX ldmatrix m8n8.x4.trans\n"); return 0;
}
