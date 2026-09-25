#include <cuda_runtime.h>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    // load a[tid]*b[tid] (or 0) into shared memory, then tree-reduce and write out[0]

    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? a[tid] * b[tid] : 0.0f;
    __syncthreads();                       // everyone has loaded before we read
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();                   // finish this level before the next
    }

    if(tid == 0) {
        out[0] = sdata[0];
    }
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    int n = a.size(); // same for b
    int num_bytes = n * sizeof(float);

    // allocate device memory
    float* d_ptr_a = nullptr;
    cudaMalloc((void**)&d_ptr_a, num_bytes);
    cudaMemcpy(d_ptr_a, a.data(), num_bytes, cudaMemcpyHostToDevice);
    float* d_ptr_b = nullptr;
    cudaMalloc((void**)&d_ptr_b, num_bytes);
    cudaMemcpy(d_ptr_b, b.data(), num_bytes, cudaMemcpyHostToDevice);
    float* d_dp_ptr = nullptr;
    cudaMalloc((void**)&d_dp_ptr, sizeof(float));

    // launch kernel
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_ptr_a, d_ptr_b, d_dp_ptr, n);

    // copy result to host
    float result = 0;
    cudaMemcpy(&result, d_dp_ptr, sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup device memory
    cudaFree(d_ptr_a);
    cudaFree(d_ptr_b);
    cudaFree(d_dp_ptr);

    return result;
}
