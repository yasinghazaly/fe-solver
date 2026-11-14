"""Tests for the C++ kernels in fesolver._kernels.

Each kernel must give the same answer as the Python method it replaces.
Hand-worked 3 x 3 matrices check the C++ code exactly. The IC(0) factor
and stiffness matrix of a reduced cantilever check it on a real finite
element pattern, against both the Python methods and numpy. The last
tests check what crosses the Python-C++ boundary: the inputs are left
unchanged and every call returns a new float64 array.
"""
import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness
from fesolver.sparse import CSR
from fesolver import _kernels

STEEL = PlaneStressMaterial(209e9, 0.3)


def dense_to_csr(A, symmetric_pattern=False):
    """CSR from a dense array, storing its non-zero entries.

    With symmetric_pattern=True, (i, j) is also stored when only A(j, i) is
    non-zero. Use it for assembled stiffness matrices: round-off can leave
    about 1e-7 at (i, j) and exactly zero at (j, i), and IC(0) takes its
    pattern from the lower triangle only. Leave it off for triangular L.
    """
    A = np.asarray(A, dtype=float)
    mask = A != 0
    if symmetric_pattern:
        mask = mask | mask.T
    rows, cols = np.nonzero(mask)
    return CSR(rows, cols, A[rows, cols], A.shape[0])


def arrays(M):
    """The three CSR arrays of M, in the order the kernels take them."""
    return M.indptr, M.indices, M.data


# Hand-worked triangular solve: L y = r gives y = [2, 2, 4] and
# L^T z = y gives z = [1, 0, 1]. L x for x = [1, 2, 3] is [2, 7, 16].
L_3 = np.array([[2., 0., 0.],
                [1., 3., 0.],
                [0., 2., 4.]])
R_3 = np.array([4., 8., 20.])


@pytest.fixture(scope="module")
def fe_factor():
    """K_ff of an 8 x 2 Quad8 cantilever fixed at the left end, with its IC(0) factor.

    Returns the CSR of K_ff, L and L^T, and a fixed random vector of length n.
    """
    mesh = structured_quad8(2.0, 0.5, 8, 2)
    K = assembly_stiffness(mesh, Quad8, STEEL, 0.01)
    fixed = [d for node, (x, _) in enumerate(mesh.nodes) if np.isclose(x, 0.0)
             for d in (2 * node, 2 * node + 1)]
    free = np.setdiff1d(np.arange(mesh.n_dofs), fixed)
    K_csr = dense_to_csr(K[np.ix_(free, free)], symmetric_pattern=True)
    L = K_csr.incomplete_cholesky()
    v = np.random.default_rng(0).standard_normal(K_csr.n)
    return K_csr, L, L.transpose(), v


# Hand examples

def test_forward_solve_hand_example():
    L = dense_to_csr(L_3)
    y = _kernels.forward_solve(*arrays(L), R_3)
    assert np.allclose(y, [2., 2., 4.], rtol=1e-14, atol=0)


def test_backward_solve_hand_example():
    LT = dense_to_csr(L_3).transpose()
    z = _kernels.backward_solve(*arrays(LT), np.array([2., 2., 4.]))
    assert np.allclose(z, [1., 0., 1.], rtol=1e-14, atol=0)


def test_matvec_hand_example():
    L = dense_to_csr(L_3)
    y = _kernels.matvec(*arrays(L), np.array([1., 2., 3.]))
    assert np.allclose(y, [2., 7., 16.], rtol=1e-14, atol=0)


def test_forward_then_backward_inverts_L_L_transpose():
    L = dense_to_csr(L_3)
    y = _kernels.forward_solve(*arrays(L), R_3)
    z = _kernels.backward_solve(*arrays(L.transpose()), y)
    assert np.allclose(L_3 @ L_3.T @ z, R_3, rtol=1e-14, atol=0)


# Against the Python methods on a finite element pattern

def test_forward_solve_matches_python(fe_factor):
    _, L, _, r = fe_factor
    y_cpp = _kernels.forward_solve(*arrays(L), r)
    y_py = np.asarray(L.forward_solve(L, r))
    assert np.allclose(y_cpp, y_py, rtol=1e-13, atol=0)


def test_backward_solve_matches_python(fe_factor):
    _, L, LT, y = fe_factor
    z_cpp = _kernels.backward_solve(*arrays(LT), y)
    z_py = L.backward_solve(LT, y)
    assert np.allclose(z_cpp, z_py, rtol=1e-13, atol=0)


def test_matvec_matches_python(fe_factor):
    K, _, _, x = fe_factor
    assert np.allclose(_kernels.matvec(*arrays(K), x), K.matvec(x), rtol=1e-13, atol=0)


def test_triangular_solves_match_numpy(fe_factor):
    _, L, LT, r = fe_factor
    L_dense = L.to_dense()
    y = _kernels.forward_solve(*arrays(L), r)
    z = _kernels.backward_solve(*arrays(LT), y)
    assert np.allclose(y, np.linalg.solve(L_dense, r), rtol=1e-10, atol=0)
    assert np.allclose(z, np.linalg.solve(L_dense.T, y), rtol=1e-10, atol=0)


def test_matvec_matches_dense_product(fe_factor):
    K, _, _, x = fe_factor
    assert np.allclose(_kernels.matvec(*arrays(K), x), K.to_dense() @ x, rtol=1e-10, atol=1e-6)


# The Python-C++ boundary

def test_inputs_are_not_modified(fe_factor):
    K, L, LT, v = fe_factor
    for M, kernel in [(L, _kernels.forward_solve), (LT, _kernels.backward_solve),
                      (K, _kernels.matvec)]:
        before = [a.copy() for a in arrays(M)] + [v.copy()]
        kernel(*arrays(M), v)
        after = list(arrays(M)) + [v]
        for b, a in zip(before, after):
            assert np.array_equal(b, a)


def test_each_call_returns_a_new_float64_array(fe_factor):
    K, _, _, x = fe_factor
    y1 = _kernels.matvec(*arrays(K), x)
    y2 = _kernels.matvec(*arrays(K), x)
    assert y1.dtype == np.float64 and y1.shape == (K.n,)
    assert y1 is not y2
    assert not np.shares_memory(y1, x)


def test_keyword_arguments_work():
    L = dense_to_csr(L_3)
    y = _kernels.forward_solve(indptr=L.indptr, indices=L.indices, data=L.data, r=R_3)
    assert np.allclose(y, [2., 2., 4.], rtol=1e-14, atol=0)


def test_csr_arrays_already_have_the_kernel_dtypes(fe_factor):
    """int64 indices and float64 values mean pybind11 reads them in place.

    Any other dtype is still accepted, but pybind11 converts it first, which
    copies the array on every call and would distort the timings.
    """
    K, L, LT, _ = fe_factor
    for M in (K, L, LT):
        assert M.indptr.dtype == np.int64
        assert M.indices.dtype == np.int64
        assert M.data.dtype == np.float64