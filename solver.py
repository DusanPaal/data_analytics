import pandas as pd
import numpy as np
import scipy

def calculate_qcd_midhinge(data: np.array) -> float:
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    return (q3 - q1) / (q1 - q3)

def normalize_data(data: np.array, method="minmax") -> np.array:
    """Normalize the given data using the specified method.

    Parameters:
    -----------
    data:
    The data to be normalized.

    method: 
    The normalization method to use:
      - "minmax" for Min-Max normalization
      - "zscore" for Z-score normalization (default).

    Returns:
    --------
    The normalized data.

    Raises:
    -------
    ValueError: 
    If an invalid method is provided.
    """
    
    if method == "minmax":
        data_range = np.max(data) - np.min(data)
        return (data - np.min(data)) / data_range
    
    if method == "zscore":
        return (data - np.mean(data)) / np.std(data)
    
    raise ValueError(f"Invalid method: {method}")

def calculate_pearson(data1: np.array, data2: np.array) -> float:
    """Calculate the Pearson correlation coefficient between two datasets.

    Parameters:
    -----------
    data1: 
    The first dataset.

    data2:
    The second dataset.

    Returns:
    --------
    The Pearson correlation coefficient between the two datasets.
    """
    
    return np.corrcoef(data1, data2)[0, 1]

def print_matrix(mat) -> None:
    for row in mat:
        print("".join(f"{num:8.2f}" for num in row.flat))

def solve_linear_system(a: np.matrix, b: np.array, method='numpy') -> np.ndarray:
    """Solve a linear system of equations.

    Parameters:
    -----------
    a: 
    The matrix of coefficients.

    b:
    The vector of constants.

    method:
    The method to use for solving the linear system:
      - 'numpy' for NumPy's built-in solver (default).
      - 'scipy' for SciPy's linear algebra solver.
      - 'lu' uses LU decomposition of matrix A to solve the system.

    Returns:
    --------
    The solution to the linear system.
    """

    # verify that the matrix is not singular by checking 
    # the matrix rank instead of the determinant, as the
    # det() is more computationally expensive and prone to 
    # overflow
    if np.linalg.matrix_rank(a) < a.shape[0]:
      raise ValueError("Matrix is singular!")

    if method == 'numpy':
      return np.linalg.solve(a, b)
    
    if method=='scipy':
      return scipy.linalg.solve(a, b)
    
    if method == 'lu':
      lu, piv = scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)
      return scipy.linalg.lu_solve((lu, piv), b)
    
    raise ValueError(f"Invalid method: {method}")

def decompose_to_lu(A: np.matrix, optimize=False) -> tuple:
    """Compute LU decomposition of a matrix with partial pivoting.

    Parameters:
    -----------
    A: 
    The matrix to decompose.

    optimize:
    Whether to use optimized routines for
    the decomposition (default: False).

    Returns:
    --------
    A tuple containing the lower triangular 
    matrix (L) and the upper triangular matrix (U).
    """

    # check that all values of the matrix are finite
    if not np.all(np.isfinite(A)):
      raise ValueError("Matrix contains non-finite values!")

    if optimize:
      result = scipy.linalg.lu(
        A, permute_l=True)
    else:
      result = scipy.linalg.lu(
        A, permute_l=True, 
        overwrite_a=True, 
        check_finite=False
      )

    return result

def monte_carlo_pi(n_samples: int) -> float:
  """Estimates the value of 𝜋 by randomly placing 
  points inside a square, and counting how many fall 
  within a quarter circle inscribed in that square.
  The ratio of points inside the circle to the total 
  points can be used to estimate π.

  Parameters:
  -----------
  n_samples:
  The number of random samples to generate (default: 10000).

  Returns:
  --------
  The estimated value of π.
  """

  inside_circle = 0

  for _ in range(n_samples):
      
      x, y = np.random.uniform(0, 1, 2)

      if x**2 + y**2 <= 1:
          inside_circle += 1

  return (inside_circle / n_samples) * 4
