#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>  // Needed for OpenMP
#include <sys/time.h>  // For gettimeofday()

void initialize_matrix(int64_t** matrix, const int size) {
    // Seed the random number generator (once globally)
    srand((unsigned int) time(NULL));

    // Parallelize the loop
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Generate random number between 1 and 10
            matrix[i][j] = (rand() % 10) + 1;
        }
    }
}

// Function to multiply matrices A and B, storing the result in C
void matmul(int64_t** A, int64_t** B, int64_t** C, const int size) {
    //#pragma acc parallel loop collapse(2)
  #pragma omp parallel for collapse(2)  
  for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int size = 500;

    if (argc > 1) {
        size = atoi(argv[1]);
        if (size <= 0) {
            fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
            return 1;
        }
    }

    // Allocate memory for matrices A, B, and C
    int64_t** A = (int64_t**) malloc(size * sizeof(int64_t*));
    int64_t** C = (int64_t**) malloc(size * sizeof(int64_t*));

    for (int i = 0; i < size; i++) {
        A[i] = (int64_t*) malloc(size * sizeof(int64_t));
        C[i] = (int64_t*) malloc(size * sizeof(int64_t));
    }

    initialize_matrix(A, size);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matmul(A, A, C, size);

    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 
                     (end.tv_usec - start.tv_usec) / 1e6;

    printf("%.0f\n", seconds);

    // Free memory
    for (int i = 0; i < size; i++) {
        free(A[i]);
        free(C[i]);
    }
    free(A);
    free(C);

    return 0;
}

