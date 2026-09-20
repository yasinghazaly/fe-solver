import numpy as np 


def solve(K,F,fixed_dofs) : 
    """Solve the constrained system K u = F for nodal displacements.

    The global stiffness matrix is singular, since an unconstrained 2D mesh
    has three rigid-body modes. Constraints are applied by partitioning:
    rows and columns belonging to fixed DOFs are removed, the reduced system
    is solved for the free DOFs, and the result is scattered back into a
    full-length vector with zeros at the supports.

    K is not modified, so reactions can be recovered afterwards as
    K @ u - F evaluated at the fixed DOFs.

    Only zero-displacement supports are handled. Prescribed non-zero
    displacements require an additional term on the right-hand side.

    Parameters
    ----------
    K : ndarray, shape (n_dofs, n_dofs)
        Global stiffness matrix from assembly.
    F : ndarray, shape (n_dofs,)
        Global load vector, zero except at loaded DOFs.
    fixed_dofs : array_like of int
        Global indices of DOFs held at zero displacement, e.g. from
        Mesh.node_dofs.

    Returns
    -------
    ndarray, shape (n_dofs,)
        Nodal displacements in metres, exactly zero at the fixed DOFs.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the reduced matrix is singular, meaning the supports do not
        remove all three rigid-body modes.
    """
    dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(dofs,fixed_dofs)
    K_partitioned = K[np.ix_(free_dofs,free_dofs)]
    F_partitioned = F[free_dofs]
    u_partitioned = np.linalg.solve(K_partitioned,F_partitioned)
    u = np.zeros(K.shape[0])
    u[free_dofs] = u_partitioned
    return u
    
    
        
    
    