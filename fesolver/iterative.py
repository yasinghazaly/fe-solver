import numpy as np 
import warnings

def cg(K,b,preconditioner = None,x0 = None,tol = 1e-8,max_iter = None) :  
    
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
    if preconditioner == "jacobi" : 
        z = M_inv * r
        
    elif preconditioner == "ic0" : 
        y = L.forward_solve(L,r)
        z = L.backward_solve(L_transposed,y)
    
    else : 
        z = r
        
    return z