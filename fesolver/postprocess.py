import numpy as np 

def element_gauss_stresses(mesh , u, element_cls , material) : 
    """Strain and stress at every Gauss point of every element.

    Each element's displacements are sliced out of the global vector by its
    DOF mapping, then strain follows from B @ u_e and stress from D @ strain.
    Results are stored per element rather than assembled: strain is a
    derivative of displacement, and derivatives are discontinuous across
    element boundaries, so neighbouring elements give genuinely different
    values at a shared node.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (n_dofs,)
        Global displacement vector from the solver.
    element_cls : type
        Element class, e.g. CST or Quad8.
    material : PlaneStressMaterial

    Returns
    -------
    strain : ndarray, shape (n_elements, n_gauss_points, 3)
    stress : ndarray, shape (n_elements, n_gauss_points, 3)
        Voigt ordering [x, y, xy]. Stress in Pa.
    """
    strain = np.zeros((mesh.n_elements,element_cls.n_gauss_points,3))
    stress = np.zeros((mesh.n_elements,element_cls.n_gauss_points,3))
    for e in range(mesh.n_elements) : 
        coords = mesh.element_coords(e)
        element = element_cls()
        strain_e, stress_e = element.gauss_point_stresses(coords,material,u[mesh.element_dofs(e)])
        strain[e] = strain_e 
        stress[e] = stress_e 
    return strain,stress

def element_nodal_stresses(mesh , stress_gp , element_cls) :
    """Extrapolate Gauss-point stresses to each element's own nodes.

    Applies the element's extrapolation matrix, which least-squares fits
    nodal values whose interpolation best reproduces the Gauss-point values.
    Still per element: a shared node appears once per element containing it,
    with a different value each time.

    Parameters
    ----------
    mesh : Mesh
    stress_gp : ndarray, shape (n_elements, n_gauss_points, 3)
        Gauss-point stresses from element_stresses.
    element_cls : type

    Returns
    -------
    ndarray, shape (n_elements, nodes_per_element, 3)
    """
    element = element_cls()
    L = element.extrapolation_matrix()
    stress_nodal = np.zeros((mesh.n_elements,element_cls.n_nodes,3))
    for i , e in enumerate(stress_gp) : 
        stress_nodal[i] = L @ e
    return stress_nodal        

def average_nodal_stresses(mesh,stress_nodal) : 
    """Average the competing nodal stresses at each global node.

    A node shared by several elements receives a different stress from each.
    The mean gives one value per node, suitable for contour plotting. The
    spread between the averaged values is a crude discretisation error
    indicator: large jumps mean the mesh is too coarse there.

    Assumes every node belongs to at least one element, which the mesh
    validates on construction.

    Parameters
    ----------
    mesh : Mesh
    stress_nodal : ndarray, shape (n_elements, nodes_per_element, 3)

    Returns
    -------
    ndarray, shape (n_nodes, 3)
    """
    stress_sum = np.zeros((mesh.n_nodes,3))
    count = np.zeros((mesh.n_nodes))
    for e , stress_e in enumerate(stress_nodal) : 
        for index , node in enumerate(mesh.connectivity[e]) : 
            stress_sum[node] += stress_e[index]
            count[node] += 1
    stress_average = stress_sum/count.reshape(-1,1)
    return stress_average
                    
def von_mises(stress) :
    """Von Mises equivalent stress for plane stress.

    The out-of-plane component is zero, so the general expression reduces to
    sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2).

    Parameters
    ----------
    stress : ndarray, shape (n, 3)
        Voigt ordering [x, y, xy]. Works on Gauss-point or nodal stresses.

    Returns
    -------
    ndarray, shape (n,)
        Same units as the input.
    """ 
    vm = np.sqrt((stress[:,0]**2) - (stress[:,0]*stress[:,1]) + (stress[:,1]**2) + (3*(stress[:,2]**2)))
    return vm