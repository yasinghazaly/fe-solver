"""Tests for Mesh — geometry, DOF mapping, orientation and validation.

The mesh used throughout is deliberately irregular: connectivity rows are
non-monotonic, node 2 is shared by all four elements, and nodes 3 and 5
belong to one element each. A mesh with tidy sequential connectivity would
pass several of these tests even with a broken implementation.
"""

import numpy as np 
import pytest 
from fesolver.mesh import Mesh 

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

@pytest.fixture 
def mesh() : 
    return Mesh(NODES , CONNECTIVITY)

def test_count(mesh) : 
    assert mesh.n_nodes == 6 
    assert mesh.n_elements == 4
    assert mesh.nodes_per_element ==3
    assert mesh.n_dofs ==12 
    
def test_element_coords(mesh) : 
    assert np.allclose(mesh.element_coords(0) , [[-0.09 , 0.19], 
                                                 [0.05 , -0.02], 
                                                 [0.17 , 0.26]])
    assert np.allclose(mesh.element_coords(1) , [[0.05 , -0.02], 
                                                 [ 0.31,  0.04], 
                                                 [0.17 , 0.26]])
    assert np.allclose(mesh.element_coords(2) , [[ 0.31,  0.04], 
                                                 [ 0.42,  0.22], 
                                                 [0.17 , 0.26]])
    assert np.allclose(mesh.element_coords(3) , [[ 0.17 , 0.26],
                                                 [0.42,  0.22], 
                                                 [0.12,  0.44]])
        
def test_element_dofs_preserve_order(mesh) : 
    """DOF lists follow connectivity order, never numerical order.

    Elements 0 and 2 have non-monotonic rows, so a sorted or set-based
    implementation returns plausible-looking DOFs that no longer correspond
    to the rows and columns of the element stiffness matrix. The resulting
    global K is still symmetric with three rigid-body modes, so nothing
    downstream catches it.
    """
    assert np.array_equal(mesh.element_dofs(0) , [6,7,0,1,4,5])
    assert np.array_equal(mesh.element_dofs(1) , [0,1,2,3,4,5])
    assert np.array_equal(mesh.element_dofs(2) , [2,3,8,9,4,5])
    assert np.array_equal(mesh.element_dofs(3) , [4,5,8,9,10,11])
    
def test_node_dofs_preserves_caller_order(mesh) : 
    assert np.array_equal(mesh.node_dofs([4,0]), [8,9,0,1])
    assert np.array_equal(mesh.node_dofs([3,5]), [6,7,10,11])
    
def test_orientation_passes_on_valid_mesh(mesh) : 
    """A clockwise element is detected, and only that element.

    Swapping two nodes in element 2 reverses its signed area. Asserting the
    exact list rather than a non-empty one ensures the check discriminates
    rather than flagging everything.
    """
    assert mesh.check_orientation(3) == []
    
def test_orientation_flags_reversed_element() : 
    assert Mesh(NODES, [[3, 0, 2],[0, 1, 2],[1, 2, 4],[2, 4, 5]]).check_orientation(3) == [2]

def test_rejects_out_of_range_index():
    with pytest.raises(IndexError):
        Mesh(NODES, [[0, 1, 6]])
        
def test_rejects_negative_index() : 
    """Negative indices are rejected rather than silently wrapping.

    Python treats -1 as the last element, so an out-by-one generator error
    in the negative direction produces a geometrically plausible element
    attached to the wrong node, with no symptom anywhere downstream.
    """
    with pytest.raises(IndexError):
        Mesh(NODES, [[0, 1, -2]])
    
def test_rejects_repeated_node() : 
    with pytest.raises(ValueError):
        Mesh(NODES, [[0, 2, 2]])
        
def test_rejects_orphan_node() : 
    """A node belonging to no element is rejected at construction.

    Orphans do not break the solve -- they contribute two zero rows to K --
    but nodal stress averaging divides by a count of zero, putting nan into
    the output behind a warning rather than an error.
    """
    with pytest.raises(ValueError):
        Mesh(NODES, [[3, 0, 2],
                    [0, 1, 2],
                    [1, 4, 2],
                    [2, 4, 3]])