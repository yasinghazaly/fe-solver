"""Tests for CSR storage, sparse assembly and the partitioned sparse solve.

The dense assembly and the dense solve are the reference throughout: the
sparse path must reproduce them, so every comparison against them uses a
relative tolerance with no absolute floor (stiffness entries are of order
1e9, so numpy's default absolute tolerance would hide real errors).

Small hand-worked matrices check the format itself, including an empty row
and an empty last row. Meshes with shared nodes check that duplicate
contributions are summed rather than overwritten.
"""
import numpy as np
import pytest
from fesolver.mesh import Mesh, structured_quad8
from fesolver.elements.cst import CST
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness, assemble_triplets, assembly_stiffness_csr
from fesolver.loads import edge_traction
from fesolver.solver import solve, sparse_solve
from fesolver.sparse import CSR

STEEL = PlaneStressMaterial(209e9, 0.3)

# Irregular six-node, four-element CST mesh with shared nodes and
# non-monotonic connectivity, as in test_assembly.py.
CST_NODES = [[ 0.05, -0.02], [ 0.31,  0.04], [ 0.17,  0.26],
             [-0.09,  0.19], [ 0.42,  0.22], [ 0.12,  0.44]]
CST_CONN = [[3, 0, 2], [0, 1, 2], [1, 4, 2], [2, 4, 5]]

# Shuffled triplets for a 4 x 4 matrix with an empty row (row 1) and two
# positions that appear twice, (0, 0) and (2, 1).
ROWS_4 = np.array([3, 0, 2, 0, 2, 3, 0, 2])
COLS_4 = np.array([3, 0, 1, 2, 1, 0, 0, 3])
VALS_4 = np.array([7.0, 4.0, 5.0, 1.0, 3.0, 2.0, 6.0, 9.0])
DENSE_4 = np.array([[10., 0., 1., 0.],
                    [ 0., 0., 0., 0.],
                    [ 0., 8., 0., 9.],
                    [ 2., 0., 0., 7.]])

# Same idea on 5 rows: rows 1 and 2 are empty.
ROWS_5 = np.array([4, 0, 3, 0, 3, 0, 3, 4])
COLS_5 = np.array([4, 1, 0, 1, 2, 3, 0, 0])
VALS_5 = np.array([6.0, 2.0, 5.0, 3.0, 1.0, 4.0, 2.0, 7.0])

# Chain of five springs (stiffness 100, 200, 50, 100, 300) over six DOFs,
# with the triplets shuffled.
CHAIN_ROWS = np.array([0, 3, 5, 4, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 0, 2, 4, 3, 2, 5])
CHAIN_COLS = np.array([1, 2, 4, 4, 2, 3, 3, 5, 4, 0, 1, 1, 2, 2, 0, 3, 3, 4, 1, 5])
CHAIN_VALS = np.array([-100., -50., -300., 300., 200., 50., 100., -300., 100., -100.,
                       100., 200., -200., 50., 100., -50., -100., -100., -200., 300.])


def scatter_add(rows, cols, vals, n):
    """Dense matrix from triplets. np.add.at sums repeated positions; K[r, c] += v would not."""
    K = np.zeros((n, n))
    np.add.at(K, (rows, cols), vals)
    return K


@pytest.fixture(params=["cst", "quad8_2x1", "quad8_8x2"])
def case(request):
    """Mesh, element class and thickness for each element type and mesh."""
    if request.param == "cst":
        return Mesh(CST_NODES, CST_CONN), CST, 0.032
    if request.param == "quad8_2x1":
        return structured_quad8(2.0, 0.5, 2, 1), Quad8, 0.01
    return structured_quad8(2.0, 0.5, 8, 2), Quad8, 0.01


# --- triplet assembly ---------------------------------------------------

def test_triplet_arrays_have_expected_length_and_valid_indices(case):
    """Each element contributes (2 * nodes_per_element)**2 triplets, all inside the matrix."""
    mesh, element_cls, t = case
    rows, cols, vals = assemble_triplets(mesh, element_cls, STEEL, t)
    expected = mesh.n_elements * (2 * mesh.nodes_per_element) ** 2
    assert len(rows) == len(cols) == len(vals) == expected
    assert rows.min() >= 0 and rows.max() < mesh.n_dofs
    assert cols.min() >= 0 and cols.max() < mesh.n_dofs


def test_triplets_summed_equal_dense_assembly(case):
    """Summing the triplets reproduces assembly_stiffness, so the same matrix is described.

    Shared nodes give repeated positions, which only a summing scatter handles.
    """
    mesh, element_cls, t = case
    rows, cols, vals = assemble_triplets(mesh, element_cls, STEEL, t)
    K = scatter_add(rows, cols, vals, mesh.n_dofs)
    assert np.allclose(K, assembly_stiffness(mesh, element_cls, STEEL, t), rtol=1e-12, atol=0)


class Asymmetric:
    """Stand-in element with a deliberately non-symmetric 6 x 6 matrix."""
    def stiffness(self, coords, material, thickness):
        return np.arange(36, dtype=float).reshape(6, 6)


def test_triplets_are_not_transposed():
    """Row and column roles are the right way round.

    A real stiffness matrix is symmetric, so swapping them would go
    unnoticed. A non-symmetric element matrix exposes it.
    """
    mesh = Mesh(CST_NODES, CST_CONN)
    rows, cols, vals = assemble_triplets(mesh, Asymmetric, STEEL, 1.0)
    K = scatter_add(rows, cols, vals, mesh.n_dofs)
    expected = np.zeros_like(K)
    for e in range(mesh.n_elements):
        dofs = mesh.element_dofs(e)
        expected[np.ix_(dofs, dofs)] += np.arange(36, dtype=float).reshape(6, 6)
    assert np.array_equal(K, expected)
    assert not np.array_equal(K, expected.T)


# --- CSR format ---------------------------------------------------------

def test_csr_hand_example_with_empty_row_and_duplicates():
    """Unsorted triplets with repeats become sorted, merged CSR arrays."""
    K = CSR(ROWS_4, COLS_4, VALS_4, 4)
    assert np.array_equal(K.indptr, [0, 2, 2, 4, 6])
    assert np.array_equal(K.indices, [0, 2, 1, 3, 0, 3])
    assert np.allclose(K.data, [10, 1, 8, 9, 2, 7], rtol=1e-14, atol=0)
    assert K.n == 4


def test_csr_empty_rows_repeat_in_indptr():
    """An empty row leaves its start position unchanged."""
    K = CSR(ROWS_5, COLS_5, VALS_5, 5)
    assert np.array_equal(K.indptr, [0, 2, 2, 2, 4, 6])
    assert np.array_equal(K.indices, [1, 3, 0, 2, 0, 4])
    assert np.allclose(K.data, [5, 4, 7, 1, 7, 6], rtol=1e-14, atol=0)


def test_csr_empty_last_row_needs_n():
    """The matrix size comes from n, not from the largest row index present."""
    K = CSR(ROWS_5, COLS_5, VALS_5, 6)
    assert np.array_equal(K.indptr, [0, 2, 2, 2, 4, 6, 6])
    assert len(K.indptr) == 7


def test_csr_structure_invariants(case):
    """indptr starts at 0, never decreases, ends at the entry count; columns rise within a row."""
    mesh, element_cls, t = case
    K = assembly_stiffness_csr(mesh, element_cls, STEEL, t)
    assert K.indptr[0] == 0
    assert K.indptr[-1] == len(K.data) == len(K.indices)
    assert np.all(np.diff(K.indptr) >= 0)
    for i in range(K.n):
        cols = K.indices[K.indptr[i]:K.indptr[i + 1]]
        assert np.all(np.diff(cols) > 0)


def test_csr_to_dense_hand_example():
    assert np.allclose(CSR(ROWS_4, COLS_4, VALS_4, 4).to_dense(), DENSE_4, rtol=1e-14, atol=0)


def test_csr_matches_dense_assembly(case):
    """The whole chain (elements, triplets, CSR, dense) agrees with the dense assembly."""
    mesh, element_cls, t = case
    K = assembly_stiffness_csr(mesh, element_cls, STEEL, t)
    assert np.allclose(K.to_dense(), assembly_stiffness(mesh, element_cls, STEEL, t),
                       rtol=1e-12, atol=0)


def test_matvec_hand_example():
    """Worked by hand: row 1 is empty, so its product is exactly zero."""
    K = CSR(ROWS_4, COLS_4, VALS_4, 4)
    assert np.allclose(K.matvec(np.array([1.0, 2.0, 3.0, 4.0])), [13, 0, 52, 30], rtol=1e-14, atol=0)


def test_matvec_matches_dense_product(case):
    """K @ x from the CSR equals the dense product for a random vector."""
    mesh, element_cls, t = case
    K = assembly_stiffness_csr(mesh, element_cls, STEEL, t)
    x = np.random.default_rng(0).standard_normal(mesh.n_dofs)
    expected = assembly_stiffness(mesh, element_cls, STEEL, t) @ x
    assert np.allclose(K.matvec(x), expected, rtol=1e-10, atol=0)


# --- partitioned solve --------------------------------------------------

def test_sparse_solve_chain_with_prescribed_end():
    """Pulling one end of a spring chain moves it rigidly: the prescribed term is applied."""
    rows = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    cols = np.array([0, 1, 0, 1, 1, 2, 1, 2])
    vals = np.array([100., -100., -100., 100., 100., -100., -100., 100.])
    K = CSR(rows, cols, vals, 3)
    u = sparse_solve(K, np.zeros(3), [0], [0.02])
    assert np.allclose(u, [0.02, 0.02, 0.02], rtol=1e-12, atol=0)


def test_sparse_solve_unsorted_fixed_dofs_and_loads():
    """Fixed DOFs given out of order, with loads and non-zero supports, match the hand answer."""
    K = CSR(CHAIN_ROWS, CHAIN_COLS, CHAIN_VALS, 6)
    F = np.array([0.0, 10.0, 0.0, 0.0, -5.0, 0.0])
    u = sparse_solve(K, F, [4, 1], [-0.5, 0.5])
    expected = [0.5, 0.5, 5.0 / 14.0, -3.0 / 14.0, -0.5, -0.5]
    assert np.allclose(u, expected, rtol=1e-8, atol=0)
    assert u[4] == -0.5 and u[1] == 0.5


def test_sparse_solve_reactions_balance_applied_load():
    """Support forces sum to minus the applied load: physics, independent of stored numbers."""
    K = CSR(CHAIN_ROWS, CHAIN_COLS, CHAIN_VALS, 6)
    F = np.array([0.0, 10.0, 0.0, 0.0, -5.0, 0.0])
    u = sparse_solve(K, F, [4, 1], [-0.5, 0.5])
    reactions = (K.matvec(u) - F)[[4, 1]]
    assert np.isclose(reactions.sum(), -F.sum(), rtol=1e-9, atol=0)


LENGTH, HEIGHT, THICKNESS, LOAD = 2.0, 0.5, 0.01, 1000.0
I = THICKNESS * HEIGHT ** 3 / 12


def end_traction(x, y):
    yc = y - HEIGHT / 2
    return (0.0, LOAD / (2 * I) * (HEIGHT ** 2 / 4 - yc ** 2))


@pytest.fixture
def cantilever():
    """Cantilever on an 8 x 2 Quad8 grid: dense K, CSR K, load vector, left-end DOFs."""
    mesh = structured_quad8(LENGTH, HEIGHT, 8, 2)
    K_dense = assembly_stiffness(mesh, Quad8, STEEL, THICKNESS)
    K_csr = assembly_stiffness_csr(mesh, Quad8, STEEL, THICKNESS)
    F = np.zeros(mesh.n_dofs)
    for e in range(mesh.n_elements):
        row = mesh.connectivity[e]
        edge_nodes = [row[1], row[5], row[2]]
        if np.allclose(mesh.nodes[edge_nodes][:, 0], LENGTH):
            F += edge_traction(mesh, edge_nodes, end_traction, THICKNESS)
    fixed = []
    for node, (x, y) in enumerate(mesh.nodes):
        if np.isclose(x, 0.0):
            fixed += [2 * node, 2 * node + 1]
    return K_dense, K_csr, F, fixed


def test_sparse_solve_matches_dense_zero_supports(cantilever):
    K_dense, K_csr, F, fixed = cantilever
    u_dense = solve(K_dense, F, fixed)
    assert np.allclose(sparse_solve(K_csr, F, fixed), u_dense,
                       rtol=1e-8, atol=1e-8 * np.max(np.abs(u_dense)))


def test_sparse_solve_matches_dense_nonzero_supports(cantilever):
    """Arbitrary non-zero support values exercise the coupling term on a real mesh."""
    K_dense, K_csr, F, fixed = cantilever
    values = 1e-5 * np.random.default_rng(0).standard_normal(len(fixed))
    u_sparse = sparse_solve(K_csr, F, fixed, values)
    u_dense = solve(K_dense, F, fixed, values)
    assert np.allclose(u_sparse, u_dense, rtol=1e-8, atol=1e-8 * np.max(np.abs(u_dense)))
    assert np.all(u_sparse[fixed] == values)

def test_sparse_solve_mismatched_fixed_values_raise(cantilever):
    _, K_csr, F, fixed = cantilever
    with pytest.raises(ValueError):
        sparse_solve(K_csr, F, fixed, [0.0, 0.0])
        
def test_diagonal_hand_example():
    """Row 1 is empty and row 2 has no (2, 2) entry, so both give zero."""
    K = CSR(ROWS_4, COLS_4, VALS_4, 4)
    assert np.allclose(K.diagonal(), [10, 0, 0, 7], rtol=1e-14, atol=0)

def test_diagonal_matches_dense(case):
    """The diagonal equals the diagonal of the dense assembly."""
    mesh, element_cls, t = case
    K = assembly_stiffness_csr(mesh, element_cls, STEEL, t)
    expected = np.diag(assembly_stiffness(mesh, element_cls, STEEL, t))
    assert np.allclose(K.diagonal(), expected, rtol=1e-12, atol=0)

def test_diagonal_is_a_vector_of_length_n(case):
    mesh, element_cls, t = case
    K = assembly_stiffness_csr(mesh, element_cls, STEEL, t)
    assert K.diagonal().shape == (K.n,)