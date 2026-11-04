import time
import numpy as np
from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.sparse import CSR
from fesolver.assembly import assembly_stiffness_csr , assemble_triplets
from fesolver.iterative import cg 
from fesolver.solver import partition_csr
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array

GRIDS = [(16,4) , (32,8) , (64,16) , (128,32)]


def build_problem(nx,ny) : 
    mesh = structured_quad8(2.0, 0.5, nx, ny)
    K = assembly_stiffness_csr(mesh, Quad8, PlaneStressMaterial(209e9, 0.3), 0.01)
    F = np.zeros(mesh.n_dofs)
    left_nodes = [i for i, (x, y) in enumerate(mesh.nodes) if np.isclose(x, 0.0)]
    corner = [i for i, (x, y) in enumerate(mesh.nodes) if np.isclose(x, 2.0) and np.isclose(y, 0.5)][0]
    F[2 * corner + 1] = -1000.0
    fixed_dofs = [d for node in left_nodes for d in (2 * node, 2 * node + 1)]
    
    return mesh,K,F,fixed_dofs

def stage_breakdown(nx,ny) : 
    mesh = structured_quad8(2.0, 0.5, nx, ny)
    element_loop_start = time.perf_counter()
    rows,cols,vals = assemble_triplets(mesh, Quad8, PlaneStressMaterial(209e9, 0.3), 0.01)
    element_loop = time.perf_counter() - element_loop_start
    
    csr_build_start = time.perf_counter()
    K = CSR(rows,cols,vals,mesh.n_dofs)
    csr_build = time.perf_counter() - csr_build_start
    
    F = np.zeros(mesh.n_dofs)
    left_nodes = [i for i, (x, y) in enumerate(mesh.nodes) if np.isclose(x, 0.0)]
    corner = [i for i, (x, y) in enumerate(mesh.nodes) if np.isclose(x, 2.0) and np.isclose(y, 0.5)][0]
    F[2 * corner + 1] = -1000.0
    fixed_dofs = [d for node in left_nodes for d in (2 * node, 2 * node + 1)]
    
    partitioning_start = time.perf_counter()
    K_ff , F_rhs, free_dofs, prescribed = partition_csr(K,F,fixed_dofs)
    partitioning = time.perf_counter() - partitioning_start
    
    ic0_factorisation_start = time.perf_counter()
    L = K_ff.incomplete_cholesky()
    ic0_factorisation = time.perf_counter() - ic0_factorisation_start
    
    tranpose_start = time.perf_counter()
    L_transposed = L.transpose()
    tranpose = time.perf_counter() - tranpose_start
    
    x = np.ones(K_ff.n)
    start = time.perf_counter()
    for _ in range(10):
        K_ff.matvec(x)
    matvec_time = (time.perf_counter() - start) / 10
    
    start = time.perf_counter()
    
    for _ in range(10) : 
        y = L.forward_solve(L,x)
        z = L.backward_solve(L_transposed, y)
    solve_time = (time.perf_counter() - start) / 10
    
    return element_loop, csr_build, partitioning, ic0_factorisation, tranpose, matvec_time, solve_time
    
    
results = []
results2 = []
for nx,ny in GRIDS : 
    mesh, K,F, fixed_dofs = build_problem(nx,ny)
    total_dofs = mesh.n_dofs
    K_ff, F_rhs, free_dofs, prescribed = partition_csr(K,F,fixed_dofs)
    csr_memory = (K_ff.data.nbytes + K_ff.indices.nbytes + K_ff.indptr.nbytes) / 1e6
    n = K_ff.n
    
    K_scipy = csr_array((K_ff.data, K_ff.indices, K_ff.indptr), shape=(K_ff.n, K_ff.n)) 
    scipy_start = time.perf_counter()
    u_spsolve = spsolve(K_scipy,F_rhs)
    scipy = time.perf_counter() - scipy_start
    
    if K_ff.n < 10000 : 
        K_dense = K_ff.to_dense()
        dense_memory = n * n * 8 / 1e6 
        dense_start = time.perf_counter()
        u_dense = np.linalg.solve(K_dense, F_rhs)
        dense = time.perf_counter() - dense_start
        dense_error = np.linalg.norm(u_dense - u_spsolve) / np.linalg.norm(u_spsolve)
    else : 
        dense = None
        dense_error = None
        dense_memory = n * n * 8 / 1e6  
           
    cg_times = []
    u_cg = []
    cg_history = []
    for p in ["jacobi" , "ic0" , None] : 
        cg_start = time.perf_counter()
        x , history = cg(K_ff,F_rhs,preconditioner = p)
        cg_time = time.perf_counter() - cg_start
        cg_times.append(cg_time)
        u_cg.append(x)
        cg_history.append(len(history)-1)
    jacobi, ic0, no_preconditioner = cg_times
    u_jacobi , u_ic0, u_no_preconditioner = u_cg
    history_jacobi , history_ic0, history_no_preconditioner = cg_history
    
    jacobi_error = np.linalg.norm(u_jacobi - u_spsolve) / np.linalg.norm(u_spsolve)
    ic0_error = np.linalg.norm(u_ic0 - u_spsolve) / np.linalg.norm(u_spsolve)
    no_preconditioner_error = np.linalg.norm(u_no_preconditioner - u_spsolve) / np.linalg.norm(u_spsolve)
    
    results.append({
                'Grid' : f'{nx}x{ny}', 
                'Dofs' : total_dofs , 
                'Free DOFS' : K_ff.n , 
                
                'Dense Time' : dense , 
                'Scipy Time' : scipy , 
                'Jacobi Time' : jacobi , 
                'ic0 Time' : ic0 , 
                'No Preconditioner Time': no_preconditioner, 
                
                'Dense Error' : dense_error , 
                'Jacobi Error' : jacobi_error , 
                'ic0 Error' : ic0_error , 
                'No Preconditioner Error': no_preconditioner_error, 
                
                'Jacobi Iterations' : history_jacobi, 
                'ic0 Iterations' : history_ic0 , 
                'No Preconditioner Iterations': history_no_preconditioner,
                
                'Dense Memory' : dense_memory , 
                'CSR Memory' : csr_memory ,  })
    
    result = stage_breakdown(nx,ny)
    results2.append(result)
        
    
print(results2)
    
    
    


