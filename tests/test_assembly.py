"""Tests for global stiffness assembly.

Uses the irregular six-node, four-element CST mesh shared with the mesh
tests. A single element cannot expose a DOF mapping error, because there is
only one way to place it; a multi-element mesh with shared nodes and
non-monotonic connectivity can.
"""

import numpy as np 
from fesolver.mesh import Mesh 
from fesolver.elements.cst import CST 
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness
import pytest

NODES = [[ 0.05, -0.02],
         [ 0.31,  0.04],
         [ 0.17,  0.26],
         [-0.09,  0.19],
         [ 0.42,  0.22],
         [ 0.12,  0.44]]
CONNECTIVITY = [[3, 0, 2],
                [0, 1, 2],
                [1, 4, 2],
                [2, 4, 5]]
CST_COORDS = [[0,0], [0,0.1] , [-0.015,0.08]]
CST_CONNECTIVITY = [[0,1,2]]
THICKNESS = 0.032

@pytest.fixture 
def K() :
    mesh = Mesh(NODES,CONNECTIVITY)
    steel = PlaneStressMaterial(209e9,0.3)
    return assembly_stiffness(mesh , CST , steel , THICKNESS)  

def test_global_stiffness_shape(K) : 
    assert K.shape == (12,12)
    
def test_global_stiffness_symmetric(K) : 
    assert np.allclose(K,K.T)
    
def test_global_stiffness_has_three_rigid_body_modes(K):
    """The assembled matrix keeps exactly three rigid-body modes.

    This is the check most sensitive to assembly errors. A contribution
    scattered to the wrong DOFs typically leaves some deformation pattern
    unresisted -- an extra zero eigenvalue -- or makes the mesh resist a
    rigid motion, removing one. Symmetry alone would not catch either.
    """
    assert np.allclose(np.linalg.eigvalsh(K)[0:3] , 0 , atol = 1e-9 * np.linalg.eigvalsh(K)[-1])  
    assert np.linalg.eigvalsh(K)[3] / np.linalg.eigvalsh(K)[-1] > 1e-6
    
def test_single_element_mesh_matches_stiffness():
    """Assembling one element reproduces that element's stiffness exactly.

    With a single element every global DOF maps to itself, so assembly
    should be a pure copy. Any difference means the scatter step is altering
    values rather than just placing them.
    """
    steel = PlaneStressMaterial(209e9,0.3)
    K_element = CST().stiffness(CST_COORDS,steel,THICKNESS)
    mesh = Mesh(CST_COORDS,CST_CONNECTIVITY)
    K_global = assembly_stiffness(mesh,CST ,steel,THICKNESS)
    assert np.allclose(K_element ,K_global)
    
def test_sparsity_matches_shared_nodes(K):
    """Stiffness couples two nodes only if they share an element.

    Nodes 3 and 5 never appear in the same element, so their block is never
    written and must be exactly zero. Nodes 2 and 5 share element 3, so
    their block must receive a contribution. Checking both proves
    contributions land in the right places, not merely that zeros exist.

    The non-zero check asks for at least one non-zero entry rather than
    four: sharing an element guarantees coupling between the nodes, not
    that every individual term is non-zero.
    """
    assert np.all(K[6:8 , 10:12]  == 0 )
    assert np.any(K[4:6 ,10:12 ] != 0 )
    