// Standalone benchmark reference, linked only by tools/matmul_bench/run.sh.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <cstring>
extern "C" int matmul_mps_reference(const float *a, const float *b, float *c,
                                    int M, int K, int N, int iterations,
                                    double *times) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      return 1;
    printf("mps_device,%s\n", device.name.UTF8String);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> ab = [device newBufferWithBytes:a
                                           length:size_t(M) * K * 4
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [device newBufferWithBytes:b
                                           length:size_t(K) * N * 4
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> cb =
        [device newBufferWithLength:size_t(M) * N * 4
                            options:MTLResourceStorageModeShared];
    if (!queue || !ab || !bb || !cb)
      return 1;
    memset(cb.contents, 0xff, size_t(M) * N * 4);
    MPSMatrix *am = [[MPSMatrix alloc]
        initWithBuffer:ab
            descriptor:[MPSMatrixDescriptor
                           matrixDescriptorWithRows:M
                                            columns:K
                                           rowBytes:K * 4
                                           dataType:MPSDataTypeFloat32]];
    MPSMatrix *bm = [[MPSMatrix alloc]
        initWithBuffer:bb
            descriptor:[MPSMatrixDescriptor
                           matrixDescriptorWithRows:K
                                            columns:N
                                           rowBytes:N * 4
                                           dataType:MPSDataTypeFloat32]];
    MPSMatrix *cm = [[MPSMatrix alloc]
        initWithBuffer:cb
            descriptor:[MPSMatrixDescriptor
                           matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 4
                                           dataType:MPSDataTypeFloat32]];
    MPSMatrixMultiplication *op =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:NO
                                         transposeRight:NO
                                             resultRows:M
                                          resultColumns:N
                                        interiorColumns:K
                                                  alpha:1
                                                   beta:0];
    for (int i = -2; i < iterations; i++) {
      auto start = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> command = [queue commandBuffer];
      [op encodeToCommandBuffer:command
                     leftMatrix:am
                    rightMatrix:bm
                   resultMatrix:cm];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "MPS failed: %s\n",
                command.error.description.UTF8String);
        return 1;
      }
      if (i >= 0)
        times[i] = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count();
    }
    memcpy(c, cb.contents, size_t(M) * N * 4);
  }
  return 0;
}
