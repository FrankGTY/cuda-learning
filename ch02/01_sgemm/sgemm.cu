#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_functions.h> // for benchmark purpose

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! C = alpha * A * B + beta * C
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param C          matrix C as provided to device
//! @param M          height of matrix A and matrix C
//! @param N          width of matrix B and matrix C
//! @param K          width of matrix A and height of matrix B
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with C
////////////////////////////////////////////////////////////////////////////////

__global__ void sgemm_gpu_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void sgemm_gpu(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 dimGrid(M / dimBlock.x,  N / dimBlock.y);
    sgemm_gpu_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

void random_init(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

void performance_estimation(void (*sgemm)(const float *, const float *, float *, int, int, int, float, float), 
                            const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    
    int test_iterations = 100;

    // create timer
    StopWatchInterface *timer = NULL;

    // inital start an operation as a warmup
    sgemm(A, B, C, M, N, K, alpha, beta);

    // record the start event
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // operation body
    for (int i = 0; i < test_iterations; ++i) {
        sgemm(A, B, C, M, N, K, alpha, beta);
    }

    // waits for GPU operation finish and recored the time
    sdkStopTimer(&timer);

    // compute and print the performance
    float operation_time = sdkGetTimerValue(&timer);
    float operation_time_1_epoch = operation_time / test_iterations;

    printf("Operation time: %.4f ms\n", operation_time_1_epoch);

    // delete timer
    sdkDeleteTimer(&timer);
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int M, N, K;
    float alpha = 2.f, beta = 1.f;
    M = N = K = 2048;

    // allocate of linear memory space
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));

    // allocate of linear gpu memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // initialize randomized values
    random_init(A, M * K);
    random_init(B, K * N);
    random_init(C, M * N);

    // copy data from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    performance_estimation(sgemm_gpu, d_A, d_B, d_C, M, N, K, alpha, beta);

    // free memory
    free(A), free(B), free(C);
    cudaFree(d_A), cudaFree(d_B), cudaFree(d_C);

    return 0;
}