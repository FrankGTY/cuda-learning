#include<stdio.h>
#include<stdlib.h>

#define N 2048
#define BLOCK_SIZE 32 

__global__ void matrix_transpose_naive(int *input, int *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * N + x;
    int idx_T = x * N + y;
    output[idx_T] = input[idx];
}

__global__ void matrix_transpose_shared(int *intput, int *output) {
    __shared__ int sharedMemory [BLOCK_SIZE][BLOCK_SIZE + 1]; // use padding +1 to avoid bank conflicts
    
    // global index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // transposed global index
    int x_T = blockIdx.y * blockDim.x + threadIdx.x;
    int y_T = blockIdx.x * blockDim.y + threadIdx.y;

    // local index
    int localIdxX = threadIdx.x;
    int localIdxY = threadIdx.y;

    int idx = y * N + x;
    int idx_T = y_T * N + x_T;

    sharedMemory[localIdxX][localIdxY] = intput[idx];
    
    __syncthreads();

    output[idx_T] = sharedMemory[localIdxY][localIdxX];
}

void fill_array(int *data) {
    for (int i = 0; i<(N*N); i++) {
        data[i] = i;
    }
}

void print_output(int *a, int *b) {
    printf("\n Original Matrix::\n");
    for (int idx = 0; idx<(N*N); idx++) {
        if (idx % N == 0) {
            printf("\n");
        }
        printf("%d ", a[idx]);
    }
    printf("\n Transposed Matrix::\n");
    for (int idx = 0; idx<(N*N); idx++) {
        if (idx % N == 0) {
            printf("\n");
        }
        printf("%d ", b[idx]);
    }
}

int main() {
    int *a, *b;
    int *d_a, *d_b;

    int size = N * N * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    fill_array(a);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(N/BLOCK_SIZE, N/BLOCK_SIZE, 1);

    matrix_transpose_naive<<<dimGrid, dimBlock>>>(d_a, d_b);
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
    print_output(a, b);
    
    matrix_transpose_shared<<<dimGrid, dimBlock>>>(d_a, d_b);
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
    print_output(a, b);

    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}