/**
 * @file    main.hpp
 * @author  Robert Pellegrin
 * @brief   main driver to test Matrix class.
 * @version 0.1
 * @date    2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "Matrix.hpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  // Default matrix size
  int size = 500;

  // If an argument is provided, parse it as matrix size
  if (argc > 1) {
    size = std::atoi(argv[1]);
    if (size <= 0) {
      std::cerr << "Invalid matrix size: " << argv[1] << endl;
      return 1;
    }
  }

  Matrix A(size, size);

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();

  auto PLU = A.mul_omp(A);

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();

  // Total duration
  auto duration = end - start;

  // Convert to minutes (truncates to whole minutes)
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

  // Convert to seconds and subtract to get the remainder
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(  //
      duration - minutes                                            //
  );

  cout << "Elapsed time: " << minutes.count() << " minutes and "
       << seconds.count() << " seconds" << endl;

  return 0;
}
