"""Tests for the L2 and energy error norms.

Two kinds of check. With a zero finite element solution the error is just
the norm of the exact field, which can be worked out by hand. With nodal
values sampled from a known field, the error is pure interpolation error:
zero for anything the element represents exactly, and shrinking at a known
rate otherwise.
"""

import numpy as np
import pytest
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.errors import l2_error, energy_error


def constant(x, y):
    return (1.0, 0.0)


def constant_strain(x, y):
    return (1e-3, 0.0, 0.0)


def quadratic(x, y):
    return (x**2, x * y)


def quadratic_strain(x, y):
    return (2 * x, x, y)


def cubic(x, y):
    return (x**3, 0.0)


def cubic_strain(x, y):
    return (3 * x**2, 0.0, 0.0)


def sample(mesh, field):
    """Nodal values of a field, in DOF order [u1, v1, u2, v2, ...]."""
    return np.array([field(x, y) for x, y in mesh.nodes]).flatten()


@pytest.fixture
def steel():
    return PlaneStressMaterial(209e9, 0.3)


def test_l2_of_constant_gap_is_root_area():
    """A gap of one everywhere gives the square root of the area."""
    mesh = structured_quad8(2.0, 1.0, 1, 1)
    u = np.zeros(mesh.n_dofs)
    assert np.isclose(l2_error(mesh, u, Quad8, constant), np.sqrt(2.0))


def test_l2_does_not_depend_on_the_mesh():
    """The same field on the same domain gives the same norm on any grid.

    Checks the sum over elements and the area factor at each Gauss point.
    """
    mesh = structured_quad8(2.0, 1.0, 4, 2)
    u = np.zeros(mesh.n_dofs)
    assert np.isclose(l2_error(mesh, u, Quad8, constant), np.sqrt(2.0))


def test_l2_is_zero_for_a_quadratic_field():
    """Quad8 represents any quadratic field exactly, so the error vanishes."""
    mesh = structured_quad8(2.0, 1.0, 4, 2)
    u = sample(mesh, quadratic)
    assert l2_error(mesh, u, Quad8, quadratic) < 1e-12


def test_l2_interpolation_error_falls_as_h_cubed():
    """Halving the element size divides the L2 error by eight.

    A cubic field cannot be represented exactly. For quadratic elements the
    displacement error scales with the cube of the element size, the rate
    the convergence study should reproduce.
    """
    coarse = structured_quad8(2.0, 1.0, 2, 1)
    fine = structured_quad8(2.0, 1.0, 4, 2)
    e_coarse = l2_error(coarse, sample(coarse, cubic), Quad8, cubic)
    e_fine = l2_error(fine, sample(fine, cubic), Quad8, cubic)
    assert np.isclose(e_coarse / e_fine, 8.0, rtol=1e-6)


def test_energy_of_constant_strain_gap(steel):
    """A uniform strain gap gives sqrt(gap * D * gap * area)."""
    mesh = structured_quad8(2.0, 1.0, 1, 1)
    u = np.zeros(mesh.n_dofs)
    expected = np.sqrt(steel.D[0, 0] * 1e-6 * 2.0)
    assert np.isclose(energy_error(mesh, u, Quad8, steel, constant_strain), expected)


def test_energy_does_not_depend_on_the_mesh(steel):
    mesh = structured_quad8(2.0, 1.0, 4, 2)
    u = np.zeros(mesh.n_dofs)
    expected = np.sqrt(steel.D[0, 0] * 1e-6 * 2.0)
    assert np.isclose(energy_error(mesh, u, Quad8, steel, constant_strain), expected)


def test_energy_is_zero_for_a_quadratic_field(steel):
    """Strains of a quadratic field are recovered to roundoff.

    The tolerance is looser than for the L2 norm because the material
    stiffness, of order 1e11, sits inside the square root.
    """
    mesh = structured_quad8(2.0, 1.0, 4, 2)
    u = sample(mesh, quadratic)
    assert energy_error(mesh, u, Quad8, steel, quadratic_strain) < 1e-6


def test_energy_interpolation_error_falls_as_h_squared(steel):
    """Halving the element size divides the energy error by four.

    Strain is a derivative of displacement, so it converges one order
    slower: the square of the element size for quadratic elements.
    """
    coarse = structured_quad8(2.0, 1.0, 2, 1)
    fine = structured_quad8(2.0, 1.0, 4, 2)
    e_coarse = energy_error(coarse, sample(coarse, cubic), Quad8, steel, cubic_strain)
    e_fine = energy_error(fine, sample(fine, cubic), Quad8, steel, cubic_strain)
    assert np.isclose(e_coarse / e_fine, 4.0, rtol=1e-6)


def test_errors_are_plain_numbers(steel):
    """Both norms return a scalar, not a one-by-one array."""
    mesh = structured_quad8(2.0, 1.0, 1, 1)
    u = np.zeros(mesh.n_dofs)
    assert np.ndim(l2_error(mesh, u, Quad8, constant)) == 0
    assert np.ndim(energy_error(mesh, u, Quad8, steel, constant_strain)) == 0