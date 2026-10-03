"""Tests for the structured Quad8 grid generator.

Every case uses unequal dimensions and unequal element counts. A square
grid with nx equal to ny makes the x and y spacings identical, which hides
any mix-up between the two directions.
"""

import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness

LENGTH = 2.0
HEIGHT = 1.0
NX = 2
NY = 1
THICKNESS = 0.01


@pytest.fixture
def mesh():
    return structured_quad8(LENGTH, HEIGHT, NX, NY)


def test_node_and_element_counts(mesh):
    assert mesh.n_nodes == (2 * NX + 1) * (2 * NY + 1) - NX * NY
    assert mesh.n_elements == NX * NY
    assert mesh.nodes_per_element == 8


def test_connectivity_two_by_one(mesh):
    """Connectivity follows the Quad8 local order, not the numbering order.

    Nodes are numbered row by row, but each element lists its corners
    anticlockwise from bottom-left and then its midsides from the bottom
    edge. Nodes 2, 6 and 10 appear in both rows: the shared edge.
    """
    assert np.array_equal(mesh.connectivity, [[0, 2, 10, 8, 1, 6, 9, 5],
                                              [2, 4, 12, 10, 3, 7, 11, 6]])


def test_orientation(mesh):
    assert mesh.check_orientation(4) == []


def test_extent_matches_dimensions(mesh):
    """The grid spans exactly the requested rectangle.

    Catches spacings computed from the wrong element count or applied to
    the wrong axis, which leave the connectivity correct and the geometry
    wrong.
    """
    assert np.isclose(mesh.nodes[:, 0].min(), 0.0)
    assert np.isclose(mesh.nodes[:, 1].min(), 0.0)
    assert np.isclose(mesh.nodes[:, 0].max(), LENGTH)
    assert np.isclose(mesh.nodes[:, 1].max(), HEIGHT)


def test_midside_nodes_at_midpoints():
    """Each midside node lies halfway between the corners of its edge.

    Local node 4 sits between corners 0 and 1, node 5 between 1 and 2,
    node 6 between 2 and 3, and node 7 between 3 and 0.
    """
    mesh = structured_quad8(3.0, 1.0, 3, 2)
    for e in range(mesh.n_elements):
        coords = np.asarray(mesh.element_coords(e))
        for k in range(4):
            midpoint = 0.5 * (coords[k] + coords[(k + 1) % 4])
            assert np.allclose(coords[4 + k], midpoint)


def test_neighbours_share_three_nodes(mesh):
    """Adjacent elements share one full edge: two corners and a midside.

    Sharing node numbers is what joins elements together. Fewer than three
    would leave the elements disconnected along that edge.
    """
    shared = np.intersect1d(mesh.connectivity[0], mesh.connectivity[1])
    assert len(shared) == 3


def test_assembled_stiffness_has_three_rigid_body_modes():
    """A generated mesh assembles into a properly connected structure.

    An extra zero eigenvalue would mean part of the grid is not joined to
    the rest; a missing one would mean distorted or inverted elements.
    """
    mesh = structured_quad8(2.0, 0.5, 4, 2)
    steel = PlaneStressMaterial(209e9, 0.3)
    K = assembly_stiffness(mesh, Quad8, steel, THICKNESS)
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(np.abs(eigenvalues[:3]) < 1e-9 * eigenvalues[-1])
    assert eigenvalues[3] / eigenvalues[-1] > 1e-6