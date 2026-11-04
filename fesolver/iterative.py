"""Preconditioned conjugate gradient solver for sparse symmetric systems."""
import numpy as np 
import warnings

def cg(K,b,preconditioner = None,x0 = None,tol = 1e-8,max_iter = None) :  
    """Solve K x = b by the preconditioned conjugate gradient method.
 
    K must be symmetric positive definite, as the reduced stiffness matrix
    K_ff is once the supports remove every rigid-body mode. Each iteration
    needs one matrix-vector product and one application of the
    preconditioner, so K is never factorised or stored densely.
 
    The preconditioner is set up once before the loop. Jacobi takes the
    inverse of the diagonal of K. IC(0) builds the incomplete Cholesky
    factor L and its transpose, and is then applied as a forward solve
    with L followed by a backward solve with L^T. M = L L^T is never formed.
 
    The iteration stops when the relative residual |r| / |b| falls to tol.
 
    Parameters
    ----------
    K : CSR
        Symmetric positive definite matrix.
    b : ndarray, shape (n,)
        Right-hand side. Must not be all zeros.
    preconditioner : {None, "jacobi", "ic0"}, optional
        None gives plain CG. Defaults to None.
    x0 : ndarray, shape (n,), optional
        Starting guess. Defaults to zero. It is not modified.
    tol : float, optional
        Target relative residual. Defaults to 1e-8.
    max_iter : int, optional
        Largest number of iterations. Defaults to 10 * n.
 
    Returns
    -------
    x : ndarray, shape (n,)
        The solution.
    history : ndarray
        Relative residual at the start and after each iteration, so its
        length is the number of iterations plus one. With x0 = 0 the first
        entry is 1.
 
    Raises
    ------
    ValueError
        If b is all zeros; for "jacobi", if the diagonal has the wrong
        length or any entry that is zero, negative or NaN; for "ic0", if
        the factorisation breaks down (a zero, negative or NaN diagonal
        entry of L).
 
    Warns
    -----
    UserWarning
        If max_iter is reached and the final relative residual is still
        above tol.
    """
    if x0 is None : 
        x0 = np.zeros(len(b))
    if max_iter is None : 
        max_iter = 10 * len(b) 
        
    if preconditioner == "jacobi" : 
        L = None 
        L_transposed = None
        M = K.diagonal()
        if len(M) != len(b): 
            raise ValueError(
                f"M has {len(M)} entries but b has {len(b)}. M must hold one diagonal "
                f"entry per free DOF, so check that it was taken from the same reduced "
                f"matrix K_ff that b belongs to."
            )
        M = np.asarray(M)
        
        bad = np.where(~(M > 0))[0]
        if len(bad) > 0:
            raise ValueError(
            f"Jacobi preconditioner needs a strictly positive diagonal, but {len(bad)} "
            f"entries are zero, negative or NaN (first rows: {bad[:5]}). "
            f"A symmetric positive definite matrix cannot have these, so check that the "
            f"boundary conditions were applied and that K_ff was assembled correctly."
        )
        M_inv = 1/M
           
    elif preconditioner == "ic0" : 
        M = None
        M_inv = None
        L = K.incomplete_cholesky()
        L_diagonal = L.data[L.indptr[1:] - 1]
        bad = np.where(~(L_diagonal > 0))[0]
        if len(bad) > 0:
            raise ValueError(
                f"IC(0) broke down: {len(bad)} diagonal entries of L are zero, "
                f"negative or NaN (first rows: {bad[:5]}). This happens when a value "
                f"under the square root goes negative. Check that K_ff is symmetric "
                f"positive definite, or use the Jacobi preconditioner instead."
            )
        L_transposed = L.transpose()
        
    else : 
        L = None 
        L_transposed = None
        M = None
        M_inv = None
        
        
    b_norm = np.linalg.norm(b)
    if b_norm == 0 : 
        raise ValueError(
            f"Right-hand side b is all zeros, so the relative residual is undefined. "
            f"Check that the reduced load vector is not empty.")
        
    x = x0.copy()
    r = b - K.matvec(x0)
    z = apply_preconditioner(L,L_transposed,r,M_inv,preconditioner)
    d = z.copy()
    
    relative_residual_norm = (np.linalg.norm(r) / b_norm)
    history = [relative_residual_norm]
    iterations = 0 
    while  relative_residual_norm > tol and iterations < max_iter : 
        K_d = K.matvec(d)
        alpha = (r.T @ z) / (d.T @ K_d)
        x = x + (alpha * d)
        r_new = r - (alpha * K_d)
        z_new = apply_preconditioner(L,L_transposed,r_new,M_inv,preconditioner)
        beta = (r_new.T @ z_new) / (r.T @ z)
        r = r_new
        z = z_new
        relative_residual_norm = (np.linalg.norm(r) / b_norm )
        history.append(relative_residual_norm)
        d = z + (beta * d)
        iterations +=1 
    
    if relative_residual_norm > tol : 
        warnings.warn(
        f"CG reached max_iter = {max_iter} without converging. "
        f"Final relative residual is {history[-1]:.2e}, target tol is {tol:.2e}. "
        f"Try increasing max_iter or loosening tol.")
        
    return x, np.array(history)

def apply_preconditioner(L,L_transposed, r, M_inv, preconditioner = None) : 
    """Apply the preconditioner to a residual: return z = M^-1 r.
 
    Called once per CG iteration. Everything expensive (L, L^T and M_inv)
    is built beforehand by cg, so this step only does the cheap work.
 
    Parameters
    ----------
    L, L_transposed : CSR or None
        IC(0) factor and its transpose. Used only for "ic0".
    r : ndarray, shape (n,)
        Residual.
    M_inv : ndarray or None
        Reciprocal of the diagonal of K. Used only for "jacobi".
    preconditioner : {None, "jacobi", "ic0"}, optional
        Defaults to None.
 
    Returns
    -------
    ndarray, shape (n,)
        M_inv * r for "jacobi"; the result of solving L y = r and then
        L^T z = y for "ic0"; r itself for None.
    """
    if preconditioner == "jacobi" : 
        z = M_inv * r
    elif preconditioner == "ic0" : 
        y = L.forward_solve(L,r)
        z = L.backward_solve(L_transposed,y)
    else : 
        z = r  
    return z