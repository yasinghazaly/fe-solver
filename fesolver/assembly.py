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
    rows, cols, vals = assemble_triplets(mesh,element_cls,material,thickness)
    K = CSR(rows,cols,vals,mesh.n_dofs)
    return K 
    
        
