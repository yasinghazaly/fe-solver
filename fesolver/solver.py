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
        If the reduced stiffness matrix is singular to working precision,
        detected as a smallest-to-largest eigenvalue ratio below 1e-12.
        This means the supports leave at least one rigid-body mode free --
        typically a structure fixed at a single node, which can still
        rotate about it.

        numpy.linalg.solve does not catch this on its own. It raises only
        on an exactly zero pivot, and roundoff turns a free rigid-body mode
        into a small non-zero eigenvalue, so an unguarded solve returns an
        enormous meaningless displacement instead of failing.

    Notes
    -----
    The singularity check computes the full eigenvalue spectrum, which
    costs about as much as the solve itself. Acceptable for dense systems;
    to be replaced by a cheaper test when moving to sparse storage and
    iterative solvers.
    """
    dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(dofs,fixed_dofs)
    K_partitioned = K[np.ix_(free_dofs,free_dofs)]
    eigenvalues = np.linalg.eigvalsh(K_partitioned)
    ratio = eigenvalues[0] / eigenvalues[-1]
    if ratio < 1e-12 : 
        raise np.linalg.LinAlgError(f"Supports do not remove all rigid-body modes: smallest-to-largest "
                                    f"eigenvalue ratio of the reduced stiffness matrix is {ratio:.1e} "
                                    f"(threshold 1e-12). Check that the boundary conditions prevent "
                                    f"translation in x and y, and rotation.")
    F_partitioned = F[free_dofs]
    u_partitioned = np.linalg.solve(K_partitioned,F_partitioned)
    u = np.zeros(K.shape[0])
    u[free_dofs] = u_partitioned
    return u
    
    
        
    
    