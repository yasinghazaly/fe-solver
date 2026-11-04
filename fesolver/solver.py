import numpy as np 
from fesolver.sparse import CSR
from fesolver.iterative import cg

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
    
def sparse_solve(K,F,fixed_dofs,fixed_values = None, preconditioner = None ) : 
    """Solve the constrained system K u = F with K stored as CSR.

    Constraints are applied by partitioning, as in solve. The DOFs are
    split into free (f) and fixed (c), giving

        K_ff u_f = F_f - K_fc u_c

    where u_c holds the prescribed displacements. The fixed rows and
    columns are removed and the free DOFs are renumbered, which gives a
    smaller CSR matrix K_ff. The K_fc u_c term is zero for supports held
    at zero, and otherwise acts as a load on the free DOFs next to a moved
    support. The reduced system is solved for u_f, and the answer is
    scattered back into a full-length vector. K_ff is solved iteratively 
    by conjugate gradients (cg), never densely. An optional preconditioner 
    reduces the number of iterations: Jacobi divides by the diagonal 
    of K_ff, and IC(0) uses an incomplete Cholesky factor L, so that 
    L L^T approximates K_ff.

    K is not modified, so reactions can be recovered afterwards as
    K.matvec(u) - F at the fixed DOFs.

    Parameters
    ----------
    K : CSR
        Global stiffness matrix from assembly_stiffness_csr.
    F : ndarray, shape (n,)
        Global load vector. Entries at fixed DOFs are not used in the solve.
    fixed_dofs : array_like of int
        Global indices of the constrained DOFs, in any order.
    fixed_values : array_like of float, optional
        Prescribed displacement in metres at each fixed DOF, in the same
        order as fixed_dofs. Defaults to zero at every fixed DOF.
    preconditioner : {None, "jacobi", "ic0"}, optional
        Preconditioner for cg. None gives plain CG; "ic0" usually needs
        far fewer iterations than "jacobi". Defaults to None.

    Returns
    -------
    ndarray, shape (n,)
        Nodal displacements in metres, equal to the prescribed values at
        the fixed DOFs.
    history : ndarray
        Relative residual of cg at the start and after each iteration, so
        its length is the number of iterations plus one.

    Raises
    ------
    ValueError
        If fixed_values and fixed_dofs differ in length.
        Raises
    ValueError
        If fixed_values and fixed_dofs differ in length, or if cg rejects
        its inputs: an all-zero reduced load, a Jacobi diagonal with zero,
        negative or NaN entries, or an IC(0) breakdown.

    """
    K_ff , F_rhs , free_dofs, prescribed = partition_csr(K,F,fixed_dofs, fixed_values)
    u_f , history= cg(K = K_ff, b =  F_rhs ,preconditioner = preconditioner)
    for index,value in enumerate(free_dofs) : 
        prescribed[value] = u_f[index]
    u = prescribed
    return u , history
    
def partition_csr(K,F,fixed_dofs,fixed_values = None) : 
    if fixed_values is None :
            fixed_values = np.zeros(len(fixed_dofs))
    else : 
            fixed_values = np.array(fixed_values , dtype = float)
            if len(fixed_values) != len(fixed_dofs): 
                raise ValueError(f"fixed_values has {len(fixed_values)} entries but fixed_dofs has "
                                f"{len(fixed_dofs)}. Give one prescribed displacement per fixed DOF, "
                                f"in the same order.")
    dofs = np.arange(K.n)
    free_dofs = np.setdiff1d(dofs,fixed_dofs)
    
    new_number = np.ones(K.n , dtype= int ) *-1
    for index, value in enumerate(free_dofs) : 
        new_number[value] = index
        
    prescribed = np.zeros(K.n , dtype = float)
    for index,value in enumerate(fixed_dofs): 
        prescribed[value] = fixed_values[index]
        
    F_rhs = []
    for index,value in enumerate(new_number) : 
        if value != -1 : 
            F_rhs.append(F[index])   
                
    rows = []
    cols = []
    vals =[] 
    for row in range(K.n) : 
        start = K.indptr[row]
        end = K.indptr[row+1]
        for position in range(start,end,1) : 
            if new_number[row] != -1 : 
                if new_number[K.indices[position]] != -1 : 
                    rows.append(new_number[row])
                    cols.append(new_number[K.indices[position]])
                    vals.append(K.data[position])
                elif new_number[K.indices[position]] == -1 : 
                    F_rhs[new_number[row]] -= K.data[position] * prescribed[K.indices[position]]
                
    F_rhs = np.asarray(F_rhs)
    K_ff = CSR(np.asarray(rows,dtype=int),np.asarray(cols,dtype= int),np.asarray(vals,dtype=float),len(free_dofs))
    
    return K_ff , F_rhs, free_dofs, prescribed
    
        