#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

#define NUM_OF_THREADS 256
#define IMAGE_SIZE 1048576

// coefficients with array of structure
struct Coefficients_AOS {
    int r;
    int g;
    int b;
    int hue;
    int saturation;
    int maxVal;
    int minVal;
    int finalVal;
};

__global__ void complicatedCalculation(Coefficients_AOS* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int gray_scale = (data[i].r + data[i].g + data[i].b) / data[i].maxVal;
    int hue_sat = data[i].hue * data[i].saturation / data[i].minVal;
    data[i].finalVal = gray_scale * hue_sat;
}

void complicatedCalculation() {
    Coefficients_AOS* d_x;
    cudaMalloc(&d_x, sizeof(Coefficients_AOS) * IMAGE_SIZE);
    int num_of_blocks = IMAGE_SIZE / NUM_OF_THREADS;
    complicatedCalculation<<<num_of_blocks, NUM_OF_THREADS>>>(d_x);
    cudaFree(d_x);
}

int main(int argc, char*argv[]) {
    complicatedCalculation();
    return 0;
}