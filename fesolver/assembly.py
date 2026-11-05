from fesolver.mesh import Mesh 
from fesolver.materials import PlaneStressMaterial
from fesolver.elements.cst import CST 
from fesolver.elements.quad8 import Quad8 
from fesolver.sparse import CSR
import numpy as np 

def assembly_stiffness(mesh,element_cls,material,thickness) : 
    """Assemble the global stiffness matrix for a mesh.

    Loops over elements, builds each one from its nodal coordinates, and
    adds its stiffness matrix into the global matrix at the DOF positions
    given by the mesh. Contributions accumulate, since nodes shared between
    elements receive a term from each.

    Parameters
    ----------
    mesh : Mesh
        Supplies element coordinates and the local-to-global DOF mapping.
    element_cls : type
        Element class to instantiate, e.g. CST or Quad8. Passed as the class
        itself, not an instance. Must be consistent with the number of nodes
        per element in the mesh connectivity.
    material : PlaneStressMaterial
        Material applied to every element.
    thickness : float
        Out-of-plane thickness in metres, applied to every element.

    Returns
    -------
    ndarray, shape (n_dofs, n_dofs)
        Symmetric global stiffness matrix, singular until boundary
        conditions are applied. For a 2D mesh it has three zero eigenvalues,
        corresponding to two translations and one rotation.
    """
    K = np.zeros((mesh.n_dofs,mesh.n_dofs))
    for e in range(mesh.n_elements) : 
        coords = mesh.element_coords(e)
        element = element_cls()
        k_e = element.stiffness(coords,material,thickness)
        dofs = mesh.element_dofs(e)
        K[np.ix_(dofs,dofs)] += k_e
    return K 

def assemble_triplets(mesh,element_cls,material,thickness) : 
    """Element stiffness contributions as coordinate (COO) triplets.

    Loops over elements and records each entry of every element stiffness
    matrix as a (row, column, value) triplet at its global DOF positions.
    Nothing is summed: where two elements share a node, the same position
    appears more than once, and the duplicates are added together later by
    CSR. Together the triplets describe the same matrix as
    assembly_stiffness, without ever storing its zeros.

    Parameters
    ----------
    mesh : Mesh
        Supplies element coordinates and the local-to-global DOF mapping.
    element_cls : type
        Element class, e.g. CST or Quad8, passed as the class itself.
    material : PlaneStressMaterial
        Material applied to every element.
    thickness : float
        Out-of-plane thickness in metres.

    Returns
    -------
    rows, cols : ndarray of int, shape (n_elements * (2 * nodes_per_element)**2,)
        Global DOF indices of each entry.
    vals : ndarray of float, same shape
        Value of each entry. Entry k of the three arrays belongs together.
    """
    element = element_cls()
    rows = []
    cols = []
    vals = []
    for e in range(mesh.n_elements) : 
        coords = mesh.element_coords(e)
        k_e = element.stiffness(coords,material,thickness)
        dofs = mesh.element_dofs(e)
        row = np.repeat(dofs,len(dofs))
        col = np.tile(dofs,len(dofs))
        val = np.ravel(k_e)
        rows.append(row)
        cols.append(col)
        vals.append(val)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return rows,cols,vals

def assembly_stiffness_csr(mesh,element_cls,material,thickness) : 
    """Assemble the global stiffness matrix in CSR format.

    Same inputs as assembly_stiffness. Builds the element contributions as
    triplets and compresses them, so the zeros of the global matrix are
    never stored. Converting the result with to_dense gives
    assembly_stiffness.

    Parameters
    ----------
    mesh : Mesh
    element_cls : type
        Element class, e.g. CST or Quad8, passed as the class itself.
    material : PlaneStressMaterial
    thickness : float
        Out-of-plane thickness in metres.

    Returns
    -------
    CSR
        Symmetric global stiffness matrix of size mesh.n_dofs, singular
        until supports are applied.
    """
    rows, cols, vals = assemble_triplets(mesh,element_cls,material,thickness)
    K = CSR(rows,cols,vals,mesh.n_dofs)
    return K 
    
        
