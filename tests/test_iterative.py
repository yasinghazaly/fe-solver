"""Tests for the conjugate gradient solver and its preconditioners.

Small hand-worked systems check the algorithm itself: the number of
iterations, the residual history, the stopping rule and the input checks.
A reduced cantilever stiffness matrix checks that all three
preconditioners (none, Jacobi and IC(0)) reach the direct solution on a
real finite element problem, and that IC(0) needs fewer iterations.
"""
import warnings

import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness, assembly_stiffness_csr
from fesolver.solver import solve, sparse_solve
from fesolver.sparse import CSR
from fesolver.iterative import cg, apply_preconditioner

STEEL = PlaneStressMaterial(209e9, 0.3)
PRECONDITIONERS = [None, "jacobi", "ic0"]


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


def spring_chain(n):
    """Tridiagonal matrix with 2 on the diagonal and -1 either side."""
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


# 2 x 2 hand case. By hand: r0 = b, alpha0 = 5/14, x1 = [5/14, 5/7],
# r1 = [-3/14, 3/14], so the relative residual after one step is
# |r1| / |b| = 0.25. The second step lands on the exact answer [1/11, 7/11].
K_2 = [[4.0, 1.0], [1.0, 3.0]]
B_2 = np.array([1.0, 2.0])
X_2 = np.array([1 / 11, 7 / 11])


@pytest.fixture(scope="module")
def fe_system():
    """Reduced stiffness K_ff and load of a 16 x 4 Quad8 cantilever.

    The left end is held at zero and a point load of -1000 N acts at the
    top-right corner, giving 448 free DOFs. The dense solve is the reference.
    """
    mesh = structured_quad8(2.0, 0.5, 16, 4)
    K = assembly_stiffness(mesh, Quad8, STEEL, 0.01)
    fixed = []
    for node, (x, y) in enumerate(mesh.nodes):
        if np.isclose(x, 0.0):
            fixed += [2 * node, 2 * node + 1]
    free = np.setdiff1d(np.arange(mesh.n_dofs), fixed)
    corner = int(np.argmax(mesh.nodes[:, 0] + mesh.nodes[:, 1]))
    F = np.zeros(mesh.n_dofs)
    F[2 * corner + 1] = -1000.0
    K_ff = K[np.ix_(free, free)]
    b = F[free]
    return dense_to_csr(K_ff, symmetric_pattern=True), b, np.linalg.solve(K_ff, b)


def test_two_by_two_converges_in_two_iterations():
    """In exact arithmetic CG needs at most n iterations for an n x n system."""
    x, history = cg(dense_to_csr(K_2), B_2)
    assert np.allclose(x, X_2, rtol=1e-12, atol=0)
    assert len(history) == 3


def test_history_matches_hand_calculation():
    """Starting from zero the first entry is 1, then 0.25 after one step."""
    _, history = cg(dense_to_csr(K_2), B_2)
    assert np.allclose(history[:2], [1.0, 0.25], rtol=1e-12, atol=0)
    assert history[-1] <= 1e-8


def test_history_is_an_array_starting_at_one():
    _, history = cg(dense_to_csr(K_2), B_2)
    assert isinstance(history, np.ndarray)
    assert history[0] == pytest.approx(1.0)


@pytest.mark.parametrize("n, expected", [(6, 3), (7, 4), (10, 5)])
def test_spring_chain_needs_half_as_many_iterations(n, expected):
    """A uniform load on a uniform chain is symmetric, so it excites only
    half of the eigenvectors, and CG stops after that many iterations."""
    _, history = cg(dense_to_csr(spring_chain(n)), np.ones(n), tol=1e-10)
    assert len(history) - 1 == expected


def test_jacobi_solves_a_diagonal_matrix_in_one_iteration():
    """For a diagonal matrix the Jacobi preconditioner is the exact inverse."""
    K = dense_to_csr(np.diag([2.0, 5.0, 10.0]))
    x, history = cg(K, np.array([2.0, 5.0, 10.0]), preconditioner="jacobi")
    assert np.allclose(x, [1.0, 1.0, 1.0], rtol=1e-12, atol=0)
    assert len(history) == 2


def test_ic0_solves_a_tridiagonal_matrix_in_one_iteration():
    """A tridiagonal matrix has no fill-in, so IC(0) is the exact Cholesky
    factor and the preconditioned system is the identity."""
    n = 8
    A = spring_chain(n)
    b = np.arange(1.0, n + 1)
    x, history = cg(dense_to_csr(A), b, preconditioner="ic0")
    assert np.allclose(x, np.linalg.solve(A, b), rtol=1e-10, atol=0)
    assert len(history) == 2


def test_exact_starting_guess_needs_no_iterations():
    x, history = cg(dense_to_csr(K_2), B_2, x0=X_2.copy())
    assert np.allclose(x, X_2, rtol=1e-12, atol=0)
    assert len(history) == 1


def test_starting_guess_is_not_modified():
    x0 = np.array([0.5, 0.5])
    cg(dense_to_csr(K_2), B_2, x0=x0)
    assert np.all(x0 == [0.5, 0.5])


def test_max_iter_reached_warns():
    with pytest.warns(UserWarning, match="max_iter"):
        _, history = cg(dense_to_csr(K_2), B_2, max_iter=1)
    assert np.allclose(history, [1.0, 0.25], rtol=1e-12, atol=0)


def test_converging_on_the_last_allowed_iteration_does_not_warn():
    """The warning depends on the final residual, not the iteration count."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x, _ = cg(dense_to_csr(K_2), B_2, max_iter=2)
    assert np.allclose(x, X_2, rtol=1e-12, atol=0)


def test_looser_tolerance_stops_earlier(fe_system):
    K, b, _ = fe_system
    _, tight = cg(K, b, tol=1e-10)
    _, loose = cg(K, b, tol=1e-4)
    assert len(loose) < len(tight)
    assert loose[-1] <= 1e-4

def test_zero_right_hand_side_raises():
    with pytest.raises(ValueError, match="all zeros"):
        cg(dense_to_csr(K_2), np.zeros(2))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_jacobi_rejects_a_bad_diagonal(bad):
    K = dense_to_csr([[4.0, 1.0], [1.0, 3.0]])
    K.data[-1] = bad  # the (1, 1) entry is the last stored value
    with pytest.raises(ValueError, match="strictly positive diagonal"):
        cg(K, B_2, preconditioner="jacobi")


def test_jacobi_rejects_a_missing_diagonal_entry():
    """A row with no stored diagonal reads as zero."""
    K = dense_to_csr([[4.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="strictly positive diagonal"):
        cg(K, B_2, preconditioner="jacobi")


def test_jacobi_rejects_a_load_of_the_wrong_length():
    with pytest.raises(ValueError, match="entries but b has"):
        cg(dense_to_csr(K_2), np.ones(3), preconditioner="jacobi")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ic0_breakdown_raises():
    """[[1, 2], [2, 1]] is symmetric but not positive definite: L(1,1) would
    be the square root of 1 - 2**2, so the factorisation breaks down."""
    with pytest.raises(ValueError, match="IC\\(0\\) broke down"):
        cg(dense_to_csr([[1.0, 2.0], [2.0, 1.0]]), B_2, preconditioner="ic0")

def test_apply_without_preconditioner_returns_r():
    r = np.array([1.0, -2.0, 3.0])
    assert np.all(apply_preconditioner(None, None, r, None, None) == r)

def test_apply_jacobi_divides_by_the_diagonal():
    r = np.array([2.0, 6.0])
    M_inv = 1 / np.array([2.0, 3.0])
    assert np.allclose(apply_preconditioner(None, None, r, M_inv, "jacobi"), [1.0, 2.0])


def test_apply_ic0_solves_with_L_and_its_transpose():
    """z = (L L^T)^-1 r, done as a forward then a backward solve."""
    L = dense_to_csr([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 2.0, 4.0]])
    r = np.array([4.0, 8.0, 20.0])
    z = apply_preconditioner(L, L.transpose(), r, None, "ic0")
    assert np.allclose(z, [1.0, 0.0, 1.0], rtol=1e-14, atol=0)


@pytest.mark.parametrize("preconditioner", PRECONDITIONERS)
def test_every_preconditioner_matches_the_direct_solve(fe_system, preconditioner):
    K, b, x_direct = fe_system
    x, history = cg(K, b, preconditioner=preconditioner)
    assert history[-1] <= 1e-8
    assert np.allclose(x, x_direct, rtol=1e-6, atol=1e-6 * np.max(np.abs(x_direct)))


def test_ic0_needs_far_fewer_iterations_than_jacobi(fe_system):
    """Measured on this problem: 275 plain, 241 Jacobi, 32 IC(0).

    The bounds are loose on purpose, so the test checks the ordering and a
    large gap without depending on round-off in the exact counts."""
    K, b, _ = fe_system
    counts = {p: len(cg(K, b, preconditioner=p)[1]) - 1 for p in PRECONDITIONERS}
    assert counts["jacobi"] <= counts[None]
    assert counts["ic0"] < counts["jacobi"] / 4


def test_ic0_factor_is_valid_on_an_fe_matrix(fe_system):
    """No breakdown: every diagonal entry of L is positive and finite."""
    K, _, _ = fe_system
    L = K.incomplete_cholesky()
    diagonal = L.data[L.indptr[1:] - 1]
    assert np.all(np.isfinite(diagonal)) and np.all(diagonal > 0)


@pytest.mark.parametrize("preconditioner", PRECONDITIONERS)
def test_sparse_solve_with_each_preconditioner_matches_dense(preconditioner):
    """Non-zero supports exercise the coupling term with every preconditioner."""
    mesh = structured_quad8(2.0, 0.5, 8, 2)
    K_dense = assembly_stiffness(mesh, Quad8, STEEL, 0.01)
    K_csr = assembly_stiffness_csr(mesh, Quad8, STEEL, 0.01)
    fixed = []
    for node, (x, y) in enumerate(mesh.nodes):
        if np.isclose(x, 0.0):
            fixed += [2 * node, 2 * node + 1]
    corner = int(np.argmax(mesh.nodes[:, 0] + mesh.nodes[:, 1]))
    F = np.zeros(mesh.n_dofs)
    F[2 * corner + 1] = -1000.0
    values = 1e-5 * np.random.default_rng(1).standard_normal(len(fixed))
    u_dense = solve(K_dense, F, fixed, values)
    u_sparse, _ = sparse_solve(K_csr, F, fixed, values, preconditioner=preconditioner)
    assert np.allclose(u_sparse, u_dense, rtol=1e-6, atol=1e-6 * np.max(np.abs(u_dense)))
    assert np.all(u_sparse[fixed] == values)