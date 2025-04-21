/**
 * @file    Matrix.cpp
 * @author  Robert Pellegrin
 * @brief   Implementation file for the Matrix class.
 * @version 0.1
 * @date    2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "Matrix.hpp"

#include <iomanip>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using std::vector;

//////////////////
// Constructors //
//////////////////

/**
 * @brief This constructor creates a matrix of the specified size (rows by
 * cols) and fills it with random integer values between 0 and 9. This process
 * is done in parallel using OpenMP. The random values are generated using the
 * C++ standard library's random number generator.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 */
Matrix::Matrix(const int64_t rows, const int64_t cols)
    : rows(rows), cols(cols), matrix(rows, vector<lld>(cols, 0.0)) {
  // Prepare random number generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> val_dist(1, 9);

  // Populate the array with random values.
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      matrix[i][j] = val_dist(gen);
}

/**
 * @brief Performs LU decomposition of the matrix.
 *
 * This method decomposes the matrix into two triangular matrices:
 * L (lower triangular) and U (upper triangular). L is initialized as an
 * identity matrix, and U starts as a copy of the original matrix. The
 * decomposition is performed using Gaussian elimination.
 *
 * @return A vector of two matrices:
 *         - Element 1 is the lower triangular matrix L.
 *         - Element 2 is the upper triangular matrix U.
 *
 * @throws std::invalid_argument if the matrix is not square.
 * @throws std::runtime_error if the matrix is singular and LU decomposition
 * cannot be performed.
 *
 */

long double roundToZero(long double value) {
  const long double threshold = 1e-12;
  return (std::abs(value) < threshold) ? 0.0 : value;
}

vector<Matrix> Matrix::luDecompose() const {
  if (rows != cols)
    throw std::invalid_argument("Matrix must be square for LU decomposition.");

  const int64_t n = rows;

  Matrix L(n, n);   // Lower
  Matrix U(*this);  // Copy of A
  Matrix P(n, n);   // Permutation

  // Initialize L as zero, P as identity
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      L.matrix[i][j] = 0.0;
      P.matrix[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (int64_t k = 0; k < n; ++k) {
    // Pivoting: find the row with the max element in column k
    int64_t maxRow = k;
    long double maxVal = std::abs(U.matrix[k][k]);

    for (int64_t i = k + 1; i < n; ++i)
      if (std::abs(U.matrix[i][k]) > maxVal) {
        maxVal = std::abs(U.matrix[i][k]);
        maxRow = i;
      }

    // Check if matrix is singular
    if (std::abs(U.matrix[maxRow][k]) < 1e-12)
      throw std::runtime_error("Matrix is singular or nearly singular.");

    // Swap rows in U
    if (maxRow != k) {
      std::swap(U.matrix[k], U.matrix[maxRow]);
      std::swap(P.matrix[k], P.matrix[maxRow]);

      // Also swap the L rows before column k
      for (int64_t j = 0; j < k; ++j)
        std::swap(L.matrix[k][j], L.matrix[maxRow][j]);
    }

    // Compute multipliers and eliminate
    for (int64_t i = k + 1; i < n; ++i) {
      lld factor = roundToZero(U.matrix[i][k] / U.matrix[k][k]);
      L.matrix[i][k] = factor;

      for (int64_t j = k; j < n; ++j)
        U.matrix[i][j] -= factor * U.matrix[k][j];
    }

    // Set diagonal of L to 1
    L.matrix[k][k] = 1.0;
  }

  return {P, L, U};  // PA = LU
}

//////////////////////
// Parallel Methods //
//////////////////////

vector<Matrix> Matrix::luDecompose_opm() const {
  // Check if the matrix is square
  if (rows != cols)
    throw std::invalid_argument("Matrix must be square for LU decomposition.");

  // Create L (lower triangular) as identity matrix, U (upper triangular) as
  // a copy of the original matrix, and P (permutation) as identity matrix
  Matrix L(rows, cols);
  Matrix U(*this);
  Matrix P(rows, cols);

  // Initialize L as identity matrix, P as identity matrix
  #pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      if (i == j) {
        L.matrix[i][j] = 1.0;
        P.matrix[i][j] = 1.0;  // Identity for P initially
      } else {
        L.matrix[i][j] = 0.0;
        P.matrix[i][j] = 0.0;
      }
    }
  }

  // Perform Gaussian elimination to decompose A into L and U
  for (int64_t k = 0; k < rows; ++k) {
    // Pivoting: Find the row with the maximum element in column k
    int64_t maxRow = k;
    long double maxVal = std::abs(U.matrix[k][k]);

    #pragma omp parallel for
    for (int64_t i = k + 1; i < rows; ++i) {
      if (std::abs(U.matrix[i][k]) > maxVal) {
        maxVal = std::abs(U.matrix[i][k]);
        maxRow = i;
      }
    }

    // If the pivot element is zero, the matrix is singular
    if (std::abs(U.matrix[maxRow][k]) < 1e-12) {
      #pragma omp critical
      throw std::runtime_error(
          "Matrix is singular, LU decomposition cannot be performed.");
    }

    // Pivoting: Swap rows in U and P
    if (maxRow != k) {
      // Swap rows in U
      #pragma omp parallel for
      for (int64_t j = 0; j < cols; ++j) {
        std::swap(U.matrix[k][j], U.matrix[maxRow][j]);
      }

      // Swap rows in P
      for (int64_t j = 0; j < cols; ++j) {
        std::swap(P.matrix[k][j], P.matrix[maxRow][j]);
      }

      // Also swap the rows in L before column k (only below the diagonal)
      #pragma omp parallel for
      for (int64_t i = 0; i < k; ++i) {
        std::swap(L.matrix[k][i], L.matrix[maxRow][i]);
      }
    }

    // Gaussian elimination --Update L and U
    #pragma omp parallel for
    for (int64_t i = k + 1; i < rows; ++i) {
      const lld factor = U.matrix[i][k] / U.matrix[k][k];

      // Update row i of U
      for (int64_t j = k; j < cols; ++j) {
        U.matrix[i][j] -= factor * U.matrix[k][j];
      }

      // Update L matrix (only the elements below the diagonal)
      L.matrix[i][k] = factor;
    }
  }

  // Return the decomposed matrices P, L, and U
  return {P, L, U};
}

/**
 * @brief Friend function to multiply a single row of matrix A with matrix B.
 * This is used to simplify threading by allowing each thread to compute one
 * row of the resulting matrix.
 *
 * @param row The row of matrix A to be multiplied. This index will be used to
 *            select the appropriate row in matrix A and store the result in
 *            the corresponding row of the result matrix.
 * @param A The first matrix (left operand) in the multiplication.
 * @param B The second matrix (right operand) in the multiplication.
 * @param result The matrix to store the result of the row multiplication. This
 *               matrix will be updated in place with the computed values.
 */
void multiplyRow(const int64_t &row, const Matrix &A, const Matrix &B,
                 Matrix *result) {
  for (int64_t j = 0; j < B.getCols(); ++j) {
    result->matrix[row][j] = 0;
    for (int64_t k = 0; k < A.getCols(); ++k)
      result->matrix[row][j] += A.matrix[row][k] * B.matrix[k][j];
  }
}

/**
 * @brief Parallel matrix multiplication using threads.
 *
 * This method performs matrix multiplication in parallel by using multiple
 * threads. Each thread computes one row of the result matrix by multiplying the
 * corresponding row of the first matrix with all the columns of the second
 * matrix.
 *
 * @param other The matrix to multiply with. Its number of rows must match the
 *              number of columns of the current matrix.
 *
 * @return A new matrix containing the result of the multiplication.
 *
 * @throws std::invalid_argument if the number of columns in the current matrix
 *         does not match the number of rows in the `other` matrix.
 */
Matrix Matrix::mul_threaded(const Matrix &other) const {
  // Make sure dimensions are correct.
  if (getCols() != other.getRows())
    throw std::invalid_argument(
        "Matrix dimensions must match for multiplication.");

  // Create a matrix to hold the result.
  Matrix result(getRows(), other.getCols());

  // Array to hold thread ids as new threads are created.
  std::thread *threads = new std::thread[getRows()];

  // Create a thread for each row.
  for (int64_t i = 0; i < getRows(); ++i)
    threads[i] = std::thread(multiplyRow,      // Entry point of thread.
                             i,                // Row assigned to thread.
                             std::ref(*this),  // Matrix A.
                             std::ref(other),  // Matrix B.
                             &result);         // Matrix containing A * B.

  // Join all threads.
  for (int64_t i = 0; i < getRows(); ++i)
    threads[i].join();

  delete[] threads;

  return result;
}

/**
 * @brief Perform parallel matrix multiplication using OpenMP.
 *
 * This method multiplies the current matrix with another matrix 'other' using
 * OpenMP for parallelization. It performs matrix multiplication with the
 * standard algorithm (row-by-column) but distributes the computation across
 * multiple threads to improve performance on large matrices.
 *
 * @param other The matrix to multiply with. Its number of rows must match the
 *              number of columns of the current matrix.
 *
 * @return A new matrix containing the result of the matrix multiplication.
 *
 * @throws std::invalid_argument if the number of columns in the current matrix
 *         does not match the number of rows in other.
 */
Matrix Matrix::mul_omp(const Matrix &other) const {
  // Ensure matrix dimensions are appropriate for multiplication.
  if (cols != other.rows)
    throw std::invalid_argument("Matrix dimensions do not match!");

  Matrix result(rows, other.cols);

  #pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < other.cols; ++j) {
      lld sum = 0;

      for (int64_t k = 0; k < cols; ++k)
        sum += matrix[i][k] * other.matrix[k][j];

      result.matrix[i][j] = sum;
    }
  }
  return result;
}

///////////////
// Accessors //
///////////////

/**
 * @brief Accessor to return number of rows.
 *
 */
int64_t Matrix::getRows() const {
  return rows;
}

/**
 * @brief Accessor to return number of columns.
 *
 */
int64_t Matrix::getCols() const {
  return cols;
}

//////////////////////////
// Overloaded Operators //
//////////////////////////

/**
 * @brief Overloaded exponentiation.
 *
 * @param power
 * @return const Matrix
 *
 * @throws invalid_argument if called on a non-square matrix.
 */
Matrix Matrix::operator^(const int64_t &power) const {
  // Exponentiation may only be performed on a square matrix.
  if (rows != cols)
    throw std::invalid_argument(
        "Exponentiation invalid on a non-square matrix.");

  Matrix result = *this;

  for (int64_t i = 1; i < power; ++i)
    result = result * *this;

  return result;
}

/**
 * @brief Overloaded matrix multiplication (not parallel).
 *
 * @param other
 * @return const Matrix
 *
 * @throws invalid_arguments if called on non-square matrix.
 */
Matrix Matrix::operator*(const Matrix &other) const {
  // Ensure dimensions match for multiplication
  if (cols != other.rows)
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication.");

  // Create result matrix with correct dimensions
  Matrix result(rows, other.cols);

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < other.cols; ++j) {
      result.matrix[i][j] = 0.0;  // Initialize the element to 0
      for (int64_t k = 0; k < cols; ++k)
        result.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
    }
  }

  return result;
}

/**
 * @brief Overloaded assignment operator.
 *
 * @param other
 * @return Matrix&
 */
Matrix &Matrix::operator=(const Matrix &other) {
  // Self-assignment check
  if (this == &other)
    return *this;

  // Speed up the copying of the Matrix with OMP.
  #pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < other.rows; ++i)
    for (int64_t j = 0; j < other.cols; ++j)
      matrix[i][j] = other.matrix[i][j];

  return *this;
}

/**
 * @brief Overloaded '==' operator to compare two Matrix objects
 *
 * @param other
 * @return Matrix&
 */
bool Matrix::operator==(const Matrix &other) const {
  // Check if the dimensions (rows and cols) are the same. If dimensions differ,
  // matrices are not equal
  if (this->rows != other.rows || this->cols != other.cols)
    return false;

  // Define a small tolerance for floating-point comparisons
  const double tolerance = 1e-12;

  // Check if all elements are the same
  bool matrix_equality = true;  // Flag to indicate matrix equality

  #pragma omp parallel for collapse(2) shared(matrix_equality)
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      if (std::abs(this->matrix[i][j] - other.matrix[i][j]) > tolerance)
        #pragma omp atomic write
        matrix_equality = false;

  return matrix_equality;
}

//////////////////////
// Friend Functions //
//////////////////////

/**
 * @brief Overloaded stream out operator to neatly print the elements of a
 *        Matrix.
 *
 * This operator allows printing the entire matrix to an output stream (e.g.,
 * std::cout), where each element of the matrix is displayed in a neat,
 * formatted manner. The elements are printed row by row, with each element
 * separated by a space.
 *
 * @param stream The output stream to which the matrix will be printed
 * (typically std::cout).
 * @param matrix The Matrix object to be printed.
 * @return std::ostream& A reference to the output stream after printing the
 * matrix.
 */
std::ostream &operator<<(std::ostream &stream, const Matrix &matrix) {
  for (int64_t row = 0; row < matrix.getRows(); ++row) {
    for (int64_t col = 0; col < matrix.getCols(); ++col)
      stream << std::setw(2) << matrix.matrix[row][col];

    stream << "\n";
  }

  return stream;
}

/**
 * @brief Overloaded stream out operator to print the GMP mpq_class object.
 *
 * This operator allows printing a mpq_class object (from the GMP library) to
 * an output stream. It converts the mpq_class object to a string using its
 * get_str() function and prints it to the stream.
 *
 * @param os The output stream to which the mpq_class object will be printed.
 * @param q The mpq_class object to be printed.
 * @return std::ostream& A reference to the output stream after printing the
 * mpq_class object.
 */
std::ostream &operator<<(std::ostream &os, const mpq_class &q) {
  return os << q.get_str();
}

/**
 * @brief Overloaded stream out operator to print the GMP __mpq_struct
 * pointer.
 *
 * This operator allows printing a pointer to a __mpq_struct (which represents
 * a rational number in GMP) to an output stream. The pointer is first
 * converted into a mpq_class object and then printed to the stream.
 *
 * @param os The output stream to which the __mpq_struct pointer will be
 * printed.
 * @param q The pointer to the __mpq_struct that will be converted and
 * printed.
 * @return std::ostream& A reference to the output stream after printing the
 * __mpq_struct pointer.
 */
std::ostream &operator<<(std::ostream &os, const __mpq_struct *q) {
  // Convert the pointer to a mpq_class object and insert it into the stream.
  return os << mpq_class(q);
}
