#include <stdio.h>
#include <stdlib.h>

#define N 512


void host_add(int *a, int *b, int *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}


void fill_array(int *data) {
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100;
    }
}