#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // thread (row, col): if row<M && col<N, C[row*N+col] = sum_k A[row*K+k]*B[k*N+col]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row<M && col<N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum; 
    }
}

std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                          const std::vector<std::vector<float>>& B) {
    // dims
    int M = A.size(), K = A[0].size(), N = B[0].size();

    // memory requirement
    int A_num_bytes_row = K * sizeof(float);
    int A_num_bytes_total = A_num_bytes_row * M;
    int B_num_bytes_row = N * sizeof(float);
    int B_num_bytes_total = B_num_bytes_row * K;
    int C_num_bytes_total = B_num_bytes_row * M;
    
    // device memory allocation - A and B
    float* d_ptr_A = nullptr;
    cudaMalloc((void**)&d_ptr_A, A_num_bytes_total);
    float* d_ptr_B = nullptr;
    cudaMalloc((void**)&d_ptr_B, B_num_bytes_total);
    //// flatten A and B to row-major 1D and copy to device
    size_t offset = 0;
    for (const auto& row : A) {
        cudaMemcpy(d_ptr_A + offset, row.data(), A_num_bytes_row, cudaMemcpyHostToDevice);
        offset += K;
    }
    offset = 0;
    for (const auto& row : B) {
        cudaMemcpy(d_ptr_B + offset, row.data(), B_num_bytes_row, cudaMemcpyHostToDevice);
        offset += N;
    }

    // device memory allocation - C
    float* d_ptr_C = nullptr;
    cudaMalloc((void**)&d_ptr_C, C_num_bytes_total);

    //run the kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_ptr_A, d_ptr_B, d_ptr_C, M, K, N);

    //copy flattened C back to host
    std::vector<float> C(M * N, 0);
    cudaMemcpy(C.data(), d_ptr_C, C_num_bytes_total, cudaMemcpyDeviceToHost);

    // release device memory
    cudaFree(d_ptr_A);
    cudaFree(d_ptr_B);
    cudaFree(d_ptr_C);
    
    return C;
}
