// Vector-typed texture fetches with linear filtering. NVIDIA Warp's texture.h
// instantiates tex1D/tex2D/tex3D for float, float2 and float4; the software
// bilinear helper used to do scalar-times-vector arithmetic and could not
// instantiate the vector forms, and tex1D did not exist at all.
extern "C" __global__ void cuda_texture_vector_fetch(cudaTextureObject_t tex1,
                                                     cudaTextureObject_t tex2,
                                                     cudaTextureObject_t tex3,
                                                     float* out, float u, float v, float w) {
    const float s1 = tex1D<float>(tex1, u);
    const float2 s2 = tex1D<float2>(tex1, u);
    const float4 s4 = tex2D<float4>(tex2, u, v);
    const float4 s34 = tex3D<float4>(tex3, u, v, w);
    const int si = tex2D<int>(tex2, u, v);
    out[0] = s1 + s2.x + s2.y + s4.x + s4.w + s34.z + static_cast<float>(si);
}
