"""Regression tests — the full pipeline against validated reference values.

Each test runs the complete chain a user would run: mesh, assembly, solve,
and stress recovery. Reference values come from an earlier independent
implementation of the same problems, itself checked against a commercial
finite element package. A failure here means the solver no longer
reproduces a known-good answer, even if every unit test still passes.
"""
import pytest 
from fesolver.elements.cst import CST
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.solver import solve
from fesolver.mesh import Mesh
from fesolver.assembly import assembly_stiffness
from fesolver.postprocess import (element_gauss_stresses, element_nodal_stresses,
                                  average_nodal_stresses, von_mises)
import numpy as np 

LUG_NODES = [[-0.0150, 0.000],
             [ 0.0000, 0.000],
             [ 0.0000, 0.100],
             [-0.0300, 0.060],
             [-0.0075, 0.000],
             [ 0.0000, 0.050],
             [-0.0150, 0.080],
             [-0.0225, 0.030]]
LUG_CONNECTIVITY = [[0, 1, 2, 3, 4, 5, 6, 7]]
THICKNESS = 0.032
U_REF= np.array([[ 5.03059646e-09,  1.34913364e-06],
                  [ 0.00000000e+00,  0.00000000e+00],
                  [ 0.00000000e+00,  0.00000000e+00],
                  [ 1.31616069e-06,  6.62628356e-06],
                  [-1.52292337e-07, -1.52573064e-07],
                  [ 0.00000000e+00,  0.00000000e+00],
                  [ 8.17320742e-07,  6.82205000e-06],
                  [-9.76877543e-08,  1.48733477e-06]])
VM_REF = np.array([43.93350567e6, 20.67163553e6, 95.61033722e6, 51.30850546e6,
                   11.23801127e6, 42.93049745e6, 57.34655092e6, 24.59230153e6])
CST_COORDS = [[0,0], [0,0.1] , [-0.015,0.08]]
CST_CONNECTIVITY = [[0 , 1 , 2]]

@pytest.fixture
def steel () : 
    return PlaneStressMaterial(209e9,0.3)

@pytest.fixture
def mesh () : 
    return Mesh(LUG_NODES,LUG_CONNECTIVITY)

@pytest.fixture
def u(steel,mesh) : 
    K = assembly_stiffness(mesh,Quad8,steel,THICKNESS)
    F = np.zeros(mesh.n_dofs)
    F[mesh.node_dofs([6])[1]] = 58860
    fixed_dofs = mesh.node_dofs([1,2,5])
    u = solve(K,F,fixed_dofs)
    return u

def test_lug_displacement(u):
    """Nodal displacements of the lifting lug match the reference.

    The comparison uses a purely relative tolerance. Displacements range
    from 1e-9 to 1e-5 m, and the smallest sits below numpy's default
    absolute tolerance of 1e-8 -- with that default it would pass even if
    computed as zero.
    """
    assert np.allclose(u.reshape(8,2), U_REF , atol = 0)
    
def test_lug_von_mises(u, steel, mesh):
    """Nodal von Mises stresses match the reference, peak 95.61 MPa.

    Exercises stress recovery end to end: Gauss-point stresses,
    extrapolation to nodes, averaging, and the equivalent stress. Values
    are of order 1e7 Pa, so the default tolerances are effectively
    relative here.
    """
    strain, stress = element_gauss_stresses(mesh, u, Quad8, steel)
    nodal = element_nodal_stresses(mesh, stress, Quad8)
    avg = average_nodal_stresses(mesh, nodal)
    vm = von_mises(avg)
    assert  np.allclose(vm, VM_REF )
    
def test_cst_constant_stress(steel):
    """A single CST under the reference load gives a pure shear stress
    state.

    The two normal components are zero up to roundoff. Comparing in MPa
    puts that roundoff around 1e-14, far below the default absolute
    tolerance, while the shear component is checked relatively.
    """
    mesh = Mesh(CST_COORDS,CST_CONNECTIVITY)
    K = assembly_stiffness(mesh,CST,steel,THICKNESS)
    F = np.zeros(mesh.n_dofs)
    F[mesh.node_dofs([2])[1]] = 58860
    u = solve(K,F,mesh.node_dofs([0,1]))
    strain, stress = element_gauss_stresses(mesh, u, CST, steel)
    assert  np.allclose(stress/10**6, [[ 0 , 0  , -3.67875000e+01]])