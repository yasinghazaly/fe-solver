"""Tests for the constrained solve.

Most tests use a single CST with nodes 0 and 1 fixed and a 10 kN load in x
at node 2. The stiffness matrix is taken straight from the element rather
than through assembly, so a failure here points at the solver alone.

Prescribed non-zero displacements are checked on a two-spring chain small
enough to solve by hand, and on the CST by moving both supports together.
"""
import pytest 
from fesolver.elements.cst import CST
from fesolver.materials import PlaneStressMaterial
from fesolver.solver import solve
from fesolver.mesh import Mesh
import numpy as np 

THICKNESS = 0.032
CST_COORDS = [[0,0], [0,0.1] , [-0.015,0.08]]
CONNECTIVITY = [[0 , 1 , 2]]
SPRING_K = np.array([[ 100.0, -100.0,    0.0],
                     [-100.0,  200.0, -100.0],
                     [   0.0, -100.0,  100.0]])

@pytest.fixture 
def mesh() : 
    return Mesh(CST_COORDS,CONNECTIVITY)

@pytest.fixture 
def K() :
    steel = PlaneStressMaterial(209e9,0.3)
    K = CST().stiffness(CST_COORDS ,steel , THICKNESS) 
    return K

@pytest.fixture 
def F(mesh) : 
    F = np.zeros(mesh.n_dofs)
    F[mesh.node_dofs([2])[0]] = 10000
    return F

@pytest.fixture 
def fixed_dofs(mesh) : 
    return mesh.node_dofs([0,1])

@pytest.fixture 
def u(K , F , fixed_dofs) : 
    u = solve(K, F , fixed_dofs)
    return u 

def test_fixed_dofs_are_exactly_zero(u):
    """Supported DOFs are exactly zero, not approximately.

    These entries are assigned by the scatter step, never computed, so an
    exact comparison is correct. Anything other than exact zero means the
    free and fixed DOFs have been mixed up.
    """
    assert np.all(u[0:4]  == 0 )
    
def test_reactions_balance_applied_load(K, F, fixed_dofs, u, mesh):
    """Support reactions are in equilibrium with the applied load.

    Reactions are the fixed rows of K applied to the full displacement
    vector, minus the load at those rows. They must sum to minus the
    applied load in each direction. This checks physics rather than stored
    numbers, so it holds for any geometry or load.
    """
    reactions = K[fixed_dofs] @ u - F[fixed_dofs]
    assert np.isclose(reactions[mesh.node_dofs([0])[0]] + reactions[mesh.node_dofs([1])[0]] , -10000)
    assert np.isclose(reactions[mesh.node_dofs([0])[1]] + reactions[mesh.node_dofs([1])[1]] , 0)
    
def test_known_single_cst_solution(u):
    """Free displacements match a hand-verified solution.

    For this geometry the reduced system decouples, so u[4] is simply the
    load divided by one diagonal stiffness term, and u[5] is zero.

    The comparison uses a relative tolerance with no absolute floor:
    displacements here are of order 1e-7 m, so numpy's default absolute
    tolerance of 1e-8 would accept errors of several percent.
    """
    assert np.isclose(u[4] , 4.0819378e-07 , atol = 0)
    assert np.isclose(u[5], 0 , atol = 1e-15)
    
def test_underconstrained_system_is_detected(K, F, mesh):
    """A structure that can still move rigidly raises instead of solving.

    Fixing only node 0 blocks both translations but leaves rotation about
    that node free, so the reduced matrix keeps one near-zero eigenvalue.
    numpy.linalg.solve alone does not catch this -- it returns an enormous
    meaningless displacement -- which is why solve checks the eigenvalue
    ratio itself before solving.
    """
    fixed_dofs = mesh.node_dofs([0])
    with pytest.raises(np.linalg.LinAlgError) : 
            u = solve(K,F,fixed_dofs)
            
def test_prescribed_displacement_two_springs():
    """A prescribed end displacement is shared correctly along a chain.

    Two equal springs in series, one end held at zero and the other pulled
    to 0.02 with no applied force. The middle node must sit halfway. Its
    equation is 200*u1 = 100*0 + 100*0.02, where the right-hand side is
    exactly the coupling term the solver moves across.
    """
    u = solve(SPRING_K, np.zeros(3), [0, 2], [0.0, 0.02])
    assert np.allclose(u, [0.0, 0.01, 0.02], rtol=1e-12, atol=0)


def test_prescribed_values_are_exact(K, F, fixed_dofs):
    """Fixed DOFs hold their prescribed values exactly.

    Like zero supports, these entries are assigned by the scatter step and
    never computed, so the comparison is exact.
    """
    values = np.array([1e-6, -2e-6, 3e-6, 0.5e-6])
    u = solve(K, F, fixed_dofs, values)
    assert np.all(u[fixed_dofs] == values)


def test_moving_all_supports_translates_rigidly(K, mesh, fixed_dofs):
    """Shifting every support by the same amount moves the body rigidly.

    Nodes 0 and 1 are both moved 1 mm in x with no load applied. The free
    node must follow by the same amount and the element must stay
    unstressed. This checks the coupling term against physics: a sign
    error or a transposed block would strain the element.
    """
    shift = 0.001
    u = solve(K, np.zeros(mesh.n_dofs), fixed_dofs, [shift, 0.0, shift, 0.0])
    assert np.isclose(u[4], shift, rtol=1e-9, atol=0)
    assert np.isclose(u[5], 0.0, atol=1e-12)

    steel = PlaneStressMaterial(209e9, 0.3)
    strain, stress = CST().gauss_point_stresses(CST_COORDS, steel, u)
    assert np.allclose(stress, 0, atol=1)


def test_mismatched_fixed_values_raise(K, F, fixed_dofs):
    with pytest.raises(ValueError):
        solve(K, F, fixed_dofs, [0.0, 0.0])
