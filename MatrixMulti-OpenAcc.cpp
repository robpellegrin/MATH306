#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

using std::cout;
using std::endl;

void initialize_matrix(int64_t** matrix, const int size) {
// Parallelize the loop
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    // Create a random number generator and distribution for each thread
    std::random_device rd;  // Random device to seed the generator
    std::mt19937 gen(
        rd());  // Mersenne Twister generator seeded with random_device
    std::uniform_int_distribution<> dis(
        1, 10);  // Uniform distribution between 1 and 10

    for (int j = 0; j < size; j++)
      matrix[i][j] = dis(gen);  // Generate and assign random value
  }
}

// Function to multiply matrices A and B, storing the result in C
void matmul(int64_t** A, int64_t** B, int64_t** C, const int size) {
#pragma acc parallel loop collapse(2)  // Parallelize both outer loops
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      C[i][j] = 0.0;  // Initialize C[i][j]
      for (int k = 0; k < size; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main(int argc, char* argv[]) {
  // If an argument is provided, parse it as matrix size
  if (argc > 1) {
    const unsigned int size = std::atoi(argv[1]);
    if (size <= 0) {
      std::cerr << "Invalid matrix size: " << argv[1] << endl;
      return 1;
    }
  }

  // Allocate memory for square matrices A, B, and C
  int64_t** A = new int64_t*[size];
  int64_t** C = new int64_t*[size];

  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    A[i] = new int64_t[size];
    C[i] = new int64_t[size];
  }

  initialize_matrix(A, size);

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();

  // Perform matrix multiplication
  matmul(A, A, C, size);

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();

  // Total duration
  auto duration = end - start;

  // Convert to seconds and subtract to get the remainder
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);

  cout << seconds.count() << endl;

  // Clean up dynamically allocated memory
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    delete[] A[i];
    delete[] C[i];
  }

  delete[] A;
  delete[] C;

  return 0;
}
