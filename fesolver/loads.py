"""Equivalent nodal forces for loads distributed along element edges."""
import numpy as np 
from fesolver.mesh import Mesh
from fesolver.quadrature import gauss_1d

def edge_shape_functions(s):
    """Quadratic shape functions along a three-node edge.

    Parameters
    ----------
    s : float
        Position along the edge in natural coordinates, from -1 at the
        first end node to +1 at the second.

    Returns
    -------
    ndarray, shape (3,)
        Values for the first end node, the middle node and the second end
        node, in that order. Each is one at its own node and zero at the
        other two, and they sum to one everywhere.
    """
    return [s*(s-1) / 2 , 1-s**2 , s*(s+1) / 2]

def edge_traction(mesh, edge_nodes, traction, thickness):
    """Consistent nodal forces for a traction along one element edge.

    Each node's force is the traction weighted by that node's shape
    function and integrated along the edge with three-point Gauss
    quadrature. The result does the same work as the distributed load. A
    uniform traction is shared 1/6, 4/6, 1/6 between end, middle and end,
    not equally.

    Three Gauss points integrate the load exactly for tractions up to
    cubic variation along the edge.

    Assumes a straight edge with the middle node at its midpoint.

    Parameters
    ----------
    mesh : Mesh
    edge_nodes : sequence of 3 int
        Global node numbers along the edge, ordered end, middle, end. In a
        Quad8 connectivity row the edges are local nodes (0, 4, 1),
        (1, 5, 2), (2, 6, 3) and (3, 7, 0). Passing end, end, middle
        raises no error and gives the wrong split.
    traction : callable
        Function taking a position (x, y) in metres and returning the
        traction (tx, ty) there in pascals.
    thickness : float
        Out-of-plane thickness in metres.

    Returns
    -------
    ndarray, shape (n_dofs,)
        Global load vector, zero except at the six DOFs of the edge. Add
        the vectors from several edges to build up a loaded boundary.
    """
    edge_coords = mesh.nodes[edge_nodes]
    edge_length = np.sqrt((edge_coords[0,0] - edge_coords[2,0])**2 + (edge_coords[0,1] - edge_coords[2,1])**2 )
    gauss_points, gauss_weights = gauss_1d(3)
    F = np.zeros((mesh.n_dofs))
    dofs = mesh.node_dofs(edge_nodes)
    
    for index , s in enumerate(gauss_points) : 
        N = np.array(edge_shape_functions(s))
        x,y = N @ edge_coords
        tx, ty = traction(x,y)
        for node in range(len(edge_nodes)) : 
                F[dofs[2*node]] += gauss_weights[index] * N[node] * tx * thickness * edge_length / 2
                F[dofs[2*node + 1]] += gauss_weights[index] * N[node] * ty * thickness * edge_length / 2
                
    return F 