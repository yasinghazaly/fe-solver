"""Gauss-Legendre quadrature on the parent domain [-1, 1]."""
import numpy as np


def gauss_1d(n):
    """
    Gauss-Legendre points and weights on [-1, 1].

    Returns points (n,) in ascending order and weights (n,), which sum to 2.
    Exact for polynomials up to degree 2n - 1.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    points, weights = np.polynomial.legendre.leggauss(n)
    return points, weights


def gauss_2d(n):
    """
    Tensor-product Gauss-Legendre rule on [-1, 1] x [-1, 1].

    n is the number of points per direction, giving n**2 in total.

    Returns points (n**2, 2) with columns (xi, eta), and weights (n**2,),
    which sum to 4. Ordered with xi as the outer loop and eta as the inner:
    (xi_1, eta_1), (xi_1, eta_2), ... Element code that depends on point
    order, such as stress extrapolation to nodes, relies on this ordering.
    """
    pts, w = gauss_1d(n)
    points = np.array(np.meshgrid(pts, pts, indexing="ij")).reshape(2, -1).T
    weights = (w * w[:, None]).ravel()
    return points, weights