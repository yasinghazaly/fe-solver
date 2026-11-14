"""Tests that the "cpp" backend gives the same answers as "python".

The C++ triangular solves do exactly the Python arithmetic. The C++
matvec sums each row in order, while the Python matvec uses numpy's dot
product, which can add the same terms in a different order, so single
products differ by about 1e-16. CG amplifies such round-off over hundreds
of iterations: without IC(0) the two backends can stop one iteration
apart. The tests therefore require the first residuals to agree to
round-off, the iteration counts to agree within one, and the solutions to
agree far more closely than the solver tolerance. This is checked
for cg with each preconditioner, for apply_preconditioner on its own, and
for sparse_solve on a full constrained problem. An unknown backend name
must be rejected rather than silently running one of the two.
"""
import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness, assembly_stiffness_csr
from fesolver.solver import sparse_solve
from fesolver.sparse import CSR
from fesolver.iterative import cg, apply_preconditioner

STEEL = PlaneStressMaterial(209e9, 0.3)
PRECONDITIONERS = [None, "jacobi", "ic0"]


def dense_to_csr(A, symmetric_pattern=False):
    """CSR from a dense array, storing its non-zero entries.

    With symmetric_pattern=True, (i, j) is also stored when only A(j, i) is
    non-zero. Use it for assembled stiffness matrices: round-off can leave
    about 1e-7 at (i, j) and exactly zero at (j, i), and IC(0) takes its
    pattern from the lower triangle only.
    """
    A = np.asarray(A, dtype=float)
    mask = A != 0
    if symmetric_pattern:
        mask = mask | mask.T
    rows, cols = np.nonzero(mask)
    return CSR(rows, cols, A[rows, cols], A.shape[0])


@pytest.fixture(scope="module")
def cantilever():
    """16 x 4 Quad8 cantilever, left end held, -1000 N at the top-right corner.

    Returns the global CSR K, the load vector, the fixed DOFs, and the
    reduced K_ff (as CSR) with its load b, which has 448 free DOFs.
    """
    mesh = structured_quad8(2.0, 0.5, 16, 4)
    K_dense = assembly_stiffness(mesh, Quad8, STEEL, 0.01)
    K_csr = assembly_stiffness_csr(mesh, Quad8, STEEL, 0.01)
    fixed = [d for node, (x, _) in enumerate(mesh.nodes) if np.isclose(x, 0.0)
             for d in (2 * node, 2 * node + 1)]
    free = np.setdiff1d(np.arange(mesh.n_dofs), fixed)
    corner = int(np.argmax(mesh.nodes[:, 0] + mesh.nodes[:, 1]))
    F = np.zeros(mesh.n_dofs)
    F[2 * corner + 1] = -1000.0
    K_ff = dense_to_csr(K_dense[np.ix_(free, free)], symmetric_pattern=True)
    return K_csr, F, fixed, K_ff, F[free]


@pytest.mark.parametrize("preconditioner", PRECONDITIONERS)
def test_cg_backends_agree(cantilever, preconditioner):
    _, _, _, K_ff, b = cantilever
    x_py, hist_py = cg(K_ff, b, preconditioner=preconditioner, backend="python")
    x_cpp, hist_cpp = cg(K_ff, b, preconditioner=preconditioner, backend="cpp")
    assert abs(len(hist_cpp) - len(hist_py)) <= 1
    assert np.allclose(hist_cpp[:20], hist_py[:20], rtol=1e-12, atol=0)
    assert np.allclose(x_cpp, x_py, rtol=0, atol=1e-10 * np.max(np.abs(x_py)))


def test_python_is_the_default_backend(cantilever):
    _, _, _, K_ff, b = cantilever
    x_default, hist_default = cg(K_ff, b, preconditioner="ic0")
    x_py, hist_py = cg(K_ff, b, preconditioner="ic0", backend="python")
    assert np.array_equal(x_default, x_py)
    assert np.array_equal(hist_default, hist_py)


def test_apply_ic0_backends_agree(cantilever):
    _, _, _, K_ff, _ = cantilever
    L = K_ff.incomplete_cholesky()
    LT = L.transpose()
    r = np.random.default_rng(0).standard_normal(K_ff.n)
    z_py = apply_preconditioner(L, LT, r, None, "ic0", backend="python")
    z_cpp = apply_preconditioner(L, LT, r, None, "ic0", backend="cpp")
    assert np.allclose(z_cpp, z_py, rtol=1e-12, atol=0)


@pytest.mark.parametrize("preconditioner", PRECONDITIONERS)
def test_sparse_solve_backends_agree(cantilever, preconditioner):
    K_csr, F, fixed, _, _ = cantilever
    u_py, hist_py = sparse_solve(K_csr, F, fixed, preconditioner=preconditioner,
                                 backend="python")
    u_cpp, hist_cpp = sparse_solve(K_csr, F, fixed, preconditioner=preconditioner,
                                   backend="cpp")
    assert abs(len(hist_cpp) - len(hist_py)) <= 1
    assert np.allclose(u_cpp, u_py, rtol=0, atol=1e-10 * np.max(np.abs(u_py)))
    assert np.all(u_cpp[fixed] == 0.0)


def test_unknown_backend_raises(cantilever):
    _, _, _, K_ff, b = cantilever
    with pytest.raises(ValueError, match="backend"):
        cg(K_ff, b, preconditioner="ic0", backend="fortran")


def test_unknown_backend_raises_in_apply_preconditioner():
    L = dense_to_csr([[2.0, 0.0], [1.0, 3.0]])
    with pytest.raises(ValueError, match="backend"):
        apply_preconditioner(L, L.transpose(), np.ones(2), None, "ic0", backend="cuda")