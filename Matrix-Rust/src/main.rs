#![allow(dead_code)]

/*!
 * -----------------------------------------------------------------------------
 * Matrix Multiplication in Rust (Serial, Threaded, and Rayon Parallel Versions)
 *
 * Author      : Robert Pellegrin
 * Date        : 03-29-2025
 * Description : This program generates two random matrices with values between
 *               1 and 9, then performs matrix multiplication using:
 *               - A serial implementation
 *               - A parallel implementation using the Rayon crate
 *
 * Dependencies:
 *   - rand = "0.8"
 *   - rayon = "1.10"
 *
 * -----------------------------------------------------------------------------
 */

use std::env;
use std::time::Instant;
use rand::Rng; // For generating random numbers
use rayon::prelude::*; // For parallel iteration using Rayon

// Define a type alias for readability — a Matrix is a 2D vector of u32 values
type Matrix = Vec<Vec<u32>>;

/// Generates a matrix of given dimensions filled with random integers from 0 to 9
fn generate_matrix(rows: usize, cols: usize) -> Matrix {
    let mut rng = rand::thread_rng(); // Create a random number generator
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(1..10)).collect()) // Generate each row
        .collect() // Collect rows into the matrix
}

/// Performs matrix multiplication serially
fn serial_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len(); // Number of rows in matrix A
    let cols = b[0].len(); // Number of columns in matrix B
    let inner = b.len(); // Shared dimension between A and B (columns of A == rows of B)

    // Initialize result matrix with zeros
    let mut result = vec![vec![0; cols]; rows];

    // Perform standard triple-nested loop for matrix multiplication
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Performs matrix multiplication in parallel using Rayon
fn rayon_parallel_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let cols = b[0].len(); // Number of columns in matrix B
    let inner = b.len(); // Shared dimension

    // Use parallel iterator over rows of A
    a.par_iter()
        .map(|a_row| {
            // For each column in B, compute the dot product of a_row and the column
            (0..cols)
                .map(|j| (0..inner).map(|k| a_row[k] * b[k][j]).sum()) // Inner product
                .collect() // Collect the result into a row
        })
        .collect() // Collect all rows into the result matrix
}

/// Utility function to print a matrix in a readable format
fn print_matrix(matrix: &Matrix) {
    for row in matrix {
        println!("{:?}", row);
    }
}

/// Main function demonstrating serial and parallel matrix multiplication.
fn main() {
    // Read matrix size from command line
    let args: Vec<String> = env::args().collect();
    let size: usize = if args.len() > 1 {
        args[1].parse().expect("Please provide a valid integer for matrix size")
    } else {
        25 // default size
    };

    let a = generate_matrix(size, size);

    // Time serial multiplication
    let start = Instant::now();
    // let _result = rayon_parallel_multiply(&a, &a);
    let _result = rayon_parallel_multiply(&a, &a);
    let duration = start.elapsed();

    // Calculate total seconds with fraction
    let total_seconds = duration.as_secs_f64().round();
    println!("{:0}", total_seconds);

}
