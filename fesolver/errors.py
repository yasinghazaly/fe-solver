"""Error norms of a finite element solution against a known exact solution."""
from fesolver.quadrature import gauss_2d
import numpy as np 

def l2_error(mesh, u, element_cls, exact_displacement,n_gauss = 5):
    """L2 norm of the displacement error over the whole mesh.

    The gap between exact and finite element displacement is squared,
    integrated over every element by Gauss quadrature, and square-rooted.
    Comparing at nodes alone would not do: nodal values can be exactly
    right while the solution between them is wrong.

    A five-point rule is used, higher than for stiffness, so that
    integration error does not add to the error being measured.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (n_dofs,)
        Finite element displacements.
    element_cls : type
        Element class, e.g. Quad8.
    exact_displacement : callable
        Function taking a position (x, y) in metres and returning the
        exact displacement (ux, uy) there in metres.

    Returns
    -------
    float
        sqrt of the integral of |u_exact - u_fe|**2 over the area. No
        thickness factor is included.
    """
    element = element_cls()
    gauss_points , gauss_weights = gauss_2d(n_gauss) 
    total = 0.0
    
    for e in range(mesh.n_elements) : 
        coords = np.asarray(mesh.element_coords(e)) 
        u_e = u[mesh.element_dofs(e)].reshape(-1,2)
        for index , (xi,eta) in enumerate(gauss_points): 
            N = element.shape_functions(xi,eta)
            x,y = N @ coords
            u_fe = N @ u_e
            u_exact = np.asarray(exact_displacement(x,y))
            gap = u_exact - u_fe
            _, det_j = element.jacobian(xi,eta,coords)
            total += (gap @ gap) * gauss_weights[index] * det_j    
            
    return np.sqrt(total)        


def energy_error(mesh, u, element_cls, material, exact_strain , n_gauss = 5):
    """Energy norm of the strain error over the whole mesh.

    As l2_error, but the gap is in strain and is weighted by the material
    stiffness: gap @ D @ gap at each Gauss point. This measures the error
    in the quantity the method actually minimises, and it converges one
    order slower than the displacement error, since strain is a derivative.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (n_dofs,)
        Finite element displacements.
    element_cls : type
        Element class, e.g. Quad8.
    material : PlaneStressMaterial
    exact_strain : callable
        Function taking a position (x, y) in metres and returning the
        exact strain (eps_x, eps_y, gamma_xy) there.

    Returns
    -------
    float
        sqrt of the integral of gap @ D @ gap over the area. No thickness
        factor and no factor of one half are included; the convention only
        needs to be the same across the meshes being compared.
    """
    element = element_cls()
    gauss_points , gauss_weights = gauss_2d(n_gauss) 
    total = 0.0
    
    for e in range(mesh.n_elements) : 
        coords = np.asarray(mesh.element_coords(e)) 
        u_e = u[mesh.element_dofs(e)]
        for index , (xi,eta) in enumerate(gauss_points): 
            N = element.shape_functions(xi,eta)
            x,y = N @ coords
            B = element.B_matrix(xi,eta,coords)
            strain_fe = B @ u_e
            strain_exact = np.asarray(exact_strain(x,y))
            gap = strain_fe - strain_exact 
            _, det_j = element.jacobian(xi,eta,coords)
            total += gap @ material.D @ gap * gauss_weights[index] * det_j  
            
    return np.sqrt(total)
    

