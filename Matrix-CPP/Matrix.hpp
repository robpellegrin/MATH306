/**
 * @file    Matrix.hpp
 * @author  Robert Pellegrin
 * @brief   Specification file for the Matrix class.
 * @version 0.1
 * @date    2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstdint>
#include <iostream>
#include <vector>

using std::ostream;
using std::vector;

using lld = long double;

class Matrix {
 private:
  const int64_t rows = 0;
  const int64_t cols = 0;
  vector<vector<lld>> matrix;

 public:
  // Constructors
  Matrix() = default;                    // Default. Here for completeness.
  ~Matrix() = default;                   // Destructor. Here for completeness.
  Matrix(const Matrix &) = default;      // Copy constructor
  Matrix(const int64_t, const int64_t);  // Accepts values for rows and colums.

  // Accessors
  int64_t getCols() const;
  int64_t getRows() const;

  vector<Matrix> luDecompose() const;
  vector<Matrix> luDecompose_opm() const;

  // Parallel multiplication
  Matrix mul_omp(const Matrix &) const;       // OpenMP
  Matrix mul_threaded(const Matrix &) const;  // Standard threads

  // Operator overloads.
  bool operator==(const Matrix &) const;
  Matrix &operator=(const Matrix &);
  Matrix operator*(const Matrix &) const;  // Serial multiplication
  Matrix operator^(const int64_t &) const;

  // Friend functions.
  friend void multiplyRow(const int64_t &, const Matrix &, const Matrix &,
                          Matrix *);
  friend long double roundToZero(long double);
  friend ostream &operator<<(std::ostream &, const Matrix &);
};

#endif  // !MATRIX_HPP
