import numpy as np 

def solve(K,F,fixed_dofs , fixed_values = None) : 
    """Solve the constrained system K u = F for nodal displacements.

    The global stiffness matrix is singular, since an unconstrained 2D mesh
    has three rigid-body modes. Constraints are applied by partitioning:
    rows and columns belonging to fixed DOFs are removed, the reduced system
    is solved for the free DOFs, and the result is scattered back into a
    full-length vector with the prescribed values at the fixed DOFs.

    K is not modified, so reactions can be recovered afterwards as
    K @ u - F evaluated at the fixed DOFs.

    Fixed DOFs may be held at zero or at prescribed non-zero values. A
    known displacement acts on the free DOFs through the stiffness terms
    coupling the two sets, so those terms, multiplied by the prescribed
    values, are moved to the right-hand side. For zero supports that
    contribution vanishes.

    Parameters
    ----------
    K : ndarray, shape (n_dofs, n_dofs)
        Global stiffness matrix from assembly.
    F : ndarray, shape (n_dofs,)
        Global load vector, zero except at loaded DOFs.
    fixed_dofs : array_like of int
        Global indices of the constrained DOFs, e.g. from Mesh.node_dofs.
    fixed_values : array_like of float, optional
        Prescribed displacement in metres at each fixed DOF, in the same
        order as fixed_dofs. Defaults to zero at every fixed DOF.

    Returns
    -------
    ndarray, shape (n_dofs,)
        Nodal displacements in metres, equal to the prescribed values at
        the fixed DOFs.

    Raises
    ------
    ValueError
        If fixed_values and fixed_dofs differ in length.
    numpy.linalg.LinAlgError
    """

    if fixed_values is None :
        fixed_values = np.zeros(len(fixed_dofs))
    else : 
        fixed_values = np.array(fixed_values , dtype = float)
        if len(fixed_values) != len(fixed_dofs): 
            raise ValueError(f"fixed_values has {len(fixed_values)} entries but fixed_dofs has "
                            f"{len(fixed_dofs)}. Give one prescribed displacement per fixed DOF, "
                            f"in the same order.")
    dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(dofs,fixed_dofs)
    K_partitioned = K[np.ix_(free_dofs,free_dofs)]
    K_pull = K[np.ix_(free_dofs,fixed_dofs)]
    eigenvalues = np.linalg.eigvalsh(K_partitioned)
    ratio = eigenvalues[0] / eigenvalues[-1]
    if ratio < 1e-12 : 
        raise np.linalg.LinAlgError(f"Supports do not remove all rigid-body modes: smallest-to-largest "
                                    f"eigenvalue ratio of the reduced stiffness matrix is {ratio:.1e} "
                                    f"(threshold 1e-12). Check that the boundary conditions prevent "
                                    f"translation in x and y, and rotation.")
    F_pull = K_pull @ fixed_values
    F_partitioned = F[free_dofs] - F_pull
    u_partitioned = np.linalg.solve(K_partitioned,F_partitioned)
    u = np.zeros(K.shape[0])
    u[free_dofs] = u_partitioned
    u[fixed_dofs] = fixed_values
    return u
    
    
        
    
    