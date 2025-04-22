/*!
 * -----------------------------------------------------------------------------
 * Matrix Multiplication in Rust (Serial, Threaded, and Rayon Parallel Versions)
 *
 * Author      : Robert Pellegrin
 * Date        : 04-14-2025
 * Description :
 *
 * Dependencies:
 *   - rand = "0.8"
 *
 * -----------------------------------------------------------------------------
 */

use rand::Rng;
use std::f64;
use std::time::Instant;

type Matrix = Vec<Vec<f64>>;

static EPSILON: f64 = 1e-9;

/// Struct to hold the result of LU decomposition
#[derive(Clone)]
struct LUResult {
    p: Matrix,
    l: Matrix,
    u: Matrix,
}

/// Rounds values close to zero to exactly zero for cleaner output
fn round_to_zero(value: f64) -> f64 {
    if value.abs() < EPSILON { 0.0 } else { value }
}

/// Performs LU decomposition with partial pivoting
/// Returns matrices P, L, and U such that PA = LU
fn lu_decompose(a: &Matrix) -> Result<LUResult, String> {
    let n = a.len();
    if n == 0 || a.iter().any(|row| row.len() != n) {
        return Err("Matrix must be square.".to_string());
    }

    let mut l = vec![vec![0.0; n]; n];
    let mut u = a.clone();
    let mut p = vec![vec![0.0; n]; n];

    // Initialize P as identity matrix
    for i in 0..n {
        p[i][i] = 1.0;
    }

    for k in 0..n {
        // Find row with largest value in current column (pivoting)
        let mut max_row = k;
        let mut max_val = u[k][k].abs();
        for i in (k + 1)..n {
            if u[i][k].abs() > max_val {
                max_val = u[i][k].abs();
                max_row = i;
            }
        }

        // If pivot is too close to zero, matrix is singular
        if u[max_row][k].abs() < EPSILON {
            return Err("Matrix is singular or nearly singular.".to_string());
        }

        // Swap rows in U and P, and partially in L
        if max_row != k {
            u.swap(k, max_row);
            p.swap(k, max_row);
            l.swap(k, max_row);
        }

        // Eliminate below the pivot and store multipliers in L
        for i in (k + 1)..n {
            let factor = round_to_zero(u[i][k] / u[k][k]);
            l[i][k] = factor;
            for j in k..n {
                u[i][j] -= factor * u[k][j];
            }
        }

        // Set diagonal of L to 1
        l[k][k] = 1.0;
    }

    Ok(LUResult { p, l, u })
}

/// Multiplies two matrices
fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();

    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Checks if two matrices are approximately equal, within a given epsilon
fn matrices_approx_equal(a: &Matrix, b: &Matrix) -> bool {
    if a.len() != b.len() || a[0].len() != b[0].len() {
        return false;
    }

    for i in 0..a.len() {
        for j in 0..a[0].len() {
            if (a[i][j] - b[i][j]).abs() > EPSILON {
                return false;
            }
        }
    }
    true
}

/// Verifies that PA ≈ LU
fn verify_lu_decomposition(original: &Matrix, lu: &LUResult) -> bool {
    let pa = matrix_multiply(&lu.p, original);
    let lu_prod = matrix_multiply(&lu.l, &lu.u);

    // Debug. Output difference matrix
    if !matrices_approx_equal(&pa, &lu_prod) {
        //print_matrix("PA", &pa);
        //print_matrix("LU", &lu_prod);
        //print_matrix_diff(&pa, &lu_prod);
        return false;
    }
    true
}

/// Prints a matrix with a label
fn print_matrix(name: &str, matrix: &Matrix) {
    println!("{name}:");
    for row in matrix {
        for val in row {
            print!("{:8.3} ", val);
        }
        println!();
    }
    println!();
}

/// Generates a square matrix filled with random integers [1, 9]
fn generate_random_matrix(n: usize) -> Matrix {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            (0..n)
                .map(|_| rng.gen_range(1..=9) as f64)
                .collect::<Vec<f64>>()
        })
        .collect()
}

fn print_matrix_diff(a: &Matrix, b: &Matrix) {
    println!("Difference Matrix (PA - LU):");
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            print!("{:8.5} ", a[i][j] - b[i][j]);
        }
        println!();
    }
    println!();
}

/// Main program entry point
fn main() {
    let n = 500;
    let a = generate_random_matrix(n);
    
    let start = Instant::now();
    let result = lu_decompose(&a);
    let duration = start.elapsed();

    match result {
        Ok(_) => println!("LU decomposition took: {:.6?}", duration),
        Err(e) => println!("Error: {}", e),
    }
}
