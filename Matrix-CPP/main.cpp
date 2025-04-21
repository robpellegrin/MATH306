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

#include <iostream>

#include "Matrix.hpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  Matrix A(500, 500);
  cout << "Matrix Created!" << endl;

  cout << "LU Decomp" << endl;
  auto PLU = A.luDecompose_opm();
  cout << "LU Decomp Finished!\n";

  // Matrix PA = PLU[0] * A;
  cout << "Performing P*A\n";
  Matrix PA = PLU[0].mul_omp(A);

  cout << "Performing L*U\n";
  Matrix LU = PLU[1].mul_omp(PLU[2]);

  // cout << "PA\n\n" << PA << endl;
  // cout << "LU\n\n" << LU << endl;

  cout << "Verifying\n";
  if (PA == LU)
    cout << "PA == LU" << endl;
  else
    cout << "PA != LU" << endl;

  return 0;
}
