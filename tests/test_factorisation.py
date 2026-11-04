"""Tests for the CSR methods behind the IC(0) preconditioner.

Covers lower_triangle, incomplete_cholesky, transpose and the two
triangular solves. Hand-worked 3 x 3 matrices check each step exactly,
including a case where IC(0) must drop fill-in that exact Cholesky would
create. A reduced cantilever stiffness matrix checks the same properties
on a real finite element pattern.
"""
import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness
from fesolver.sparse import CSR

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


# Non-symmetric 4 x 4 with an empty row (row 1), for transpose.
DENSE_4 = np.array([[10., 0., 1., 0.],
                    [ 0., 0., 0., 0.],
                    [ 0., 8., 0., 9.],
                    [ 2., 0., 0., 7.]])

# Hand-worked triangular solve: L y = r gives y = [2, 2, 4] and
# L^T z = y gives z = [1, 0, 1].
L_3 = np.array([[2., 0., 0.],
                [1., 3., 0.],
                [0., 2., 4.]])
R_3 = np.array([4., 8., 20.])


@pytest.fixture(scope="module")
def fe_matrix():
    """Reduced stiffness K_ff of an 8 x 2 Quad8 cantilever fixed at the left end."""
    mesh = structured_quad8(2.0, 0.5, 8, 2)
    K = assembly_stiffness(mesh, Quad8, STEEL, 0.01)
    fixed = [d for node, (x, _) in enumerate(mesh.nodes) if np.isclose(x, 0.0)
             for d in (2 * node, 2 * node + 1)]
    free = np.setdiff1d(np.arange(mesh.n_dofs), fixed)
    K_ff = K[np.ix_(free, free)]
    return K_ff, dense_to_csr(K_ff, symmetric_pattern=True)


def test_lower_triangle_hand_example():
    K = dense_to_csr([[4., 1., 2.], [1., 5., 3.], [2., 3., 6.]])
    assert np.all(K.lower_triangle().to_dense() == [[4., 0., 0.],
                                                     [1., 5., 0.],
                                                     [2., 3., 6.]])


def test_lower_triangle_matches_numpy(fe_matrix):
    K_ff, K = fe_matrix
    assert np.all(K.lower_triangle().to_dense() == np.tril(K_ff))


def test_lower_triangle_keeps_the_diagonal_last_in_each_row(fe_matrix):
    """The triangular solves rely on this: L(i, i) sits at indptr[i + 1] - 1."""
    _, K = fe_matrix
    L = K.lower_triangle()
    assert np.all(L.indices[L.indptr[1:] - 1] == np.arange(L.n))


def test_lower_triangle_leaves_the_original_unchanged(fe_matrix):
    _, K = fe_matrix
    before = K.data.copy()
    K.lower_triangle()
    assert np.all(K.data == before)

def test_ic0_tridiagonal_is_exact():
    """No fill-in is possible, so L L^T reproduces K exactly."""
    K = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    L = dense_to_csr(K).incomplete_cholesky().to_dense()
    assert np.allclose(L, np.linalg.cholesky(K), rtol=1e-12, atol=0)
    assert np.allclose(L @ L.T, K, rtol=1e-12, atol=1e-15)


def test_ic0_full_matrix_equals_exact_cholesky():
    """A full pattern leaves nothing to drop, so IC(0) is exact Cholesky,
    here L = [[2, 0, 0], [1, 2, 0], [1, 1, 2]]."""
    K = np.array([[4., 2., 2.], [2., 5., 3.], [2., 3., 6.]])
    L = dense_to_csr(K).incomplete_cholesky().to_dense()
    assert np.allclose(L, [[2., 0., 0.], [1., 2., 0.], [1., 1., 2.]], rtol=1e-12, atol=0)


def test_ic0_drops_fill_in():
    """Exact Cholesky puts -0.129 at (2, 1), where K is zero. IC(0) keeps
    that entry at zero, so L L^T matches K only on K's own pattern."""
    K = np.array([[4., 1., 1.], [1., 4., 0.], [1., 0., 4.]])
    L = dense_to_csr(K).incomplete_cholesky().to_dense()
    assert L[2, 1] == 0.0
    assert np.linalg.cholesky(K)[2, 1] == pytest.approx(-0.129, abs=1e-3)
    assert L[2, 2] == pytest.approx(np.sqrt(4 - 0.25), rel=1e-12)
    product = L @ L.T
    pattern = K != 0
    assert np.allclose(product[pattern], K[pattern], rtol=1e-12, atol=0)
    assert not np.isclose(product[2, 1], 0.0)


def test_ic0_keeps_the_pattern_of_the_lower_triangle(fe_matrix):
    _, K = fe_matrix
    L = K.incomplete_cholesky()
    T = K.lower_triangle()
    assert np.all(L.indptr == T.indptr)
    assert np.all(L.indices == T.indices)


def test_ic0_matches_K_on_its_stored_positions(fe_matrix):
    """The defining property of IC(0): (L L^T)(i, j) = K(i, j) wherever K is stored."""
    K_ff, K = fe_matrix
    L = K.incomplete_cholesky().to_dense()
    pattern = K.to_dense() != 0
    pattern |= pattern.T
    scale = np.max(np.abs(K_ff))
    assert np.allclose((L @ L.T)[pattern], K_ff[pattern], rtol=1e-10, atol=1e-12 * scale)


def test_transpose_matches_numpy():
    """Non-symmetric, with an empty row that becomes an empty column."""
    assert np.all(dense_to_csr(DENSE_4).transpose().to_dense() == DENSE_4.T)


def test_transpose_twice_returns_the_original():
    K = dense_to_csr(DENSE_4)
    back = K.transpose().transpose()
    assert np.all(back.indptr == K.indptr)
    assert np.all(back.indices == K.indices)
    assert np.all(back.data == K.data)


def test_transpose_of_L_puts_the_diagonal_first(fe_matrix):
    """The backward solve relies on this: in L^T, L(i, i) sits at indptr[i]."""
    _, K = fe_matrix
    LT = K.incomplete_cholesky().transpose()
    assert np.all(LT.indices[LT.indptr[:-1]] == np.arange(LT.n))


def test_forward_solve_hand_example():
    L = dense_to_csr(L_3)
    assert np.allclose(L.forward_solve(L, R_3), [2., 2., 4.], rtol=1e-14, atol=0)


def test_backward_solve_hand_example():
    L = dense_to_csr(L_3)
    z = L.backward_solve(L.transpose(), np.array([2., 2., 4.]))
    assert np.allclose(z, [1., 0., 1.], rtol=1e-14, atol=0)


def test_forward_then_backward_inverts_L_L_transpose():
    L = dense_to_csr(L_3)
    z = L.backward_solve(L.transpose(), L.forward_solve(L, R_3))
    assert np.allclose(L_3 @ L_3.T @ z, R_3, rtol=1e-14, atol=0)


def test_triangular_solves_match_numpy_on_an_fe_factor(fe_matrix):
    _, K = fe_matrix
    L = K.incomplete_cholesky()
    LT = L.transpose()
    L_dense = L.to_dense()
    r = np.random.default_rng(0).standard_normal(L.n)
    y = np.asarray(L.forward_solve(L, r))
    z = L.backward_solve(LT, y)
    assert np.allclose(y, np.linalg.solve(L_dense, r), rtol=1e-10, atol=0)
    assert np.allclose(z, np.linalg.solve(L_dense.T, y), rtol=1e-10, atol=0)