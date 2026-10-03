"""Tests for consistent nodal forces from edge tractions.

The mesh is 3.0 m by 1.0 m with two elements across, so vertical edges are
1.0 m long and horizontal element edges 1.5 m. Unequal lengths make sure
the edge length is taken from the node coordinates and not assumed.
"""

import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.loads import edge_shape_functions, edge_traction

THICKNESS = 0.01
RIGHT_EDGE = [4, 7, 12]      # end, middle, end; vertical, 1.0 m long
BOTTOM_EDGE = [0, 1, 2]      # end, middle, end; horizontal, 1.5 m long


def uniform_x(x, y):
    return (1e6, 0.0)


def uniform_down(x, y):
    return (0.0, -1e6)


def rising(x, y):
    return (1e6 * y, 0.0)


def parabolic(x, y):
    return (1e6 * (1 - (2 * y - 1) ** 2), 0.0)


@pytest.fixture
def mesh():
    return structured_quad8(3.0, 1.0, 2, 1)


def test_edge_shape_functions_kronecker_delta():
    assert np.allclose(edge_shape_functions(-1), [1, 0, 0])
    assert np.allclose(edge_shape_functions(0), [0, 1, 0])
    assert np.allclose(edge_shape_functions(1), [0, 0, 1])


def test_edge_shape_functions_partition_of_unity():
    for s in [-0.83, -0.31, 0.12, 0.57, 0.94]:
        assert np.isclose(np.sum(edge_shape_functions(s)), 1.0)


def test_uniform_traction_splits_one_four_one(mesh):
    """A uniform load is shared 1/6, 4/6, 1/6 between end, middle, end.

    The middle node takes four times as much as each end, because its
    shape function covers the whole edge. An equal three-way split gives
    the right total but the wrong stresses near the loaded edge.
    """
    F = edge_traction(mesh, RIGHT_EDGE, uniform_x, THICKNESS)
    total = 1e6 * 1.0 * THICKNESS
    assert np.allclose(F[[8, 14, 24]], total * np.array([1, 4, 1]) / 6)


def test_total_force_matches_traction(mesh):
    F = edge_traction(mesh, RIGHT_EDGE, uniform_x, THICKNESS)
    assert np.isclose(F.sum(), 1e6 * 1.0 * THICKNESS)


def test_only_edge_dofs_receive_force(mesh):
    """Every entry outside the edge's six DOFs stays exactly zero.

    Those entries are never written, so the comparison is exact.
    """
    F = edge_traction(mesh, RIGHT_EDGE, uniform_x, THICKNESS)
    edge_dofs = mesh.node_dofs(RIGHT_EDGE)
    others = np.setdiff1d(np.arange(mesh.n_dofs), edge_dofs)
    assert np.all(F[others] == 0)


def test_components_do_not_mix(mesh):
    """An x traction loads only x DOFs, and a y traction only y DOFs."""
    F = edge_traction(mesh, RIGHT_EDGE, uniform_x, THICKNESS)
    assert np.all(F[[9, 15, 25]] == 0)


def test_edge_length_taken_from_coordinates(mesh):
    """A longer edge carries proportionally more force.

    The bottom edge is 1.5 m long against 1.0 m for the vertical ones, so
    a length assumed from the wrong direction shows up here.
    """
    F = edge_traction(mesh, BOTTOM_EDGE, uniform_down, THICKNESS)
    total = -1e6 * 1.5 * THICKNESS
    assert np.allclose(F[[1, 3, 5]], total * np.array([1, 4, 1]) / 6)
    assert np.all(F[[0, 2, 4]] == 0)


def test_linearly_varying_traction(mesh):
    """A load rising from zero to p along the edge splits 0, 1/3, 1/6.

    The traction is evaluated at each Gauss point's real position, so this
    fails if the position is computed wrongly. The node at the zero end
    receives nothing, although the load is non-zero along most of the edge.
    """
    F = edge_traction(mesh, RIGHT_EDGE, rising, THICKNESS)
    scale = 1e6 * 1.0 * THICKNESS
    assert np.allclose(F[[8, 14, 24]], scale * np.array([0, 1 / 3, 1 / 6]))


def test_parabolic_traction_integrates_exactly(mesh):
    """Three Gauss points integrate a parabolic load without error.

    Quadratic shape functions times a quadratic traction is a quartic,
    which a three-point rule handles exactly. The total is two thirds of
    the peak value times the edge area.
    """
    F = edge_traction(mesh, RIGHT_EDGE, parabolic, THICKNESS)
    assert np.isclose(F.sum(), (2 / 3) * 1e6 * 1.0 * THICKNESS)