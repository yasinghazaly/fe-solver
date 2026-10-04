import numpy as np 
import matplotlib.pyplot as plt 

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
        Gauss-point stresses from element_gauss_stresses.
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
    spread between the competing values is a crude discretisation error
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

def plot_deformed_shape(mesh,u,scale_factor,element_cls, save_path = None) : 
    """Draw the mesh before and after loading.

    Every node is moved by its displacement multiplied by a scale factor,
    since real displacements are far too small to see. Each element is drawn
    as a closed outline through its boundary nodes, once in its original
    position and once in its deformed position.

    Midside nodes are joined by straight segments, so curved Quad8 edges
    appear as two straight pieces.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (n_dofs,)
        Global displacement vector from the solver, in m.
    scale_factor : float
        Multiplier applied to the displacements for display only. A value
        of 0 draws the deformed shape on top of the original.
    element_cls : type
        Element class, e.g. CST or Quad8. Supplies the order in which the
        element's nodes are visited to trace its outline.
    """
    deformed_position = mesh.nodes + (scale_factor * u.reshape(-1,2))
    boundary_order = element_cls().boundary_order
    for row in mesh.connectivity : 
        original = []
        deformed = []
        for index in range(len(boundary_order)) : 
            node = row[boundary_order[index]]
            original.append(mesh.nodes[node])
            deformed.append(deformed_position[node])
        original = np.asarray(original)
        deformed = np.asarray(deformed)
        plt.plot(original[:, 0], original[:, 1],
                 color="0.7", linewidth=0.8, linestyle="--")
        plt.plot(deformed[:, 0], deformed[:, 1],
                 color="tab:blue", linewidth=1.4)
    plt.plot([], [], color="0.7", linewidth=0.8, linestyle="--", label="Original")
    plt.plot([], [], color="tab:blue", linewidth=1.4, label="Deformed")
    plt.legend(frameon=False)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Deformed shape (displacements scaled x{scale_factor})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if save_path is not None :
        plt.savefig(save_path, dpi=200)
    plt.show()

    
def plot_von_mises(mesh, vm, element_cls, save_path = None):
    """Filled contour plot of nodal von Mises stress on the undeformed mesh.

    Contours are filled across triangles, so each element is first split
    into triangles whose corners are its nodes, using the element's own
    triangle list. The colour is interpolated linearly inside each triangle,
    which is a display approximation to the element's true stress field.
    Element outlines are drawn on top.

    A perfectly constant field produces an empty plot, because contours
    need at least two distinct values.

    Parameters
    ----------
    mesh : Mesh
    vm : ndarray, shape (n_nodes,)
        Von Mises stress at each global node, in Pa. Usually von_mises
        applied to the output of average_nodal_stresses. Converted to MPa
        for display.
    element_cls : type
        Element class, e.g. CST or Quad8. Supplies the triangle split and
        the outline order.
    """
    triangles = element_cls().triangles
    boundary_order = element_cls().boundary_order
    
    all_triangles = []
    for row in mesh.connectivity:
        for triangle in triangles:
            all_triangles.append([row[triangle[0]], row[triangle[1]], row[triangle[2]]])
    all_triangles = np.asarray(all_triangles)

    contour = plt.tricontourf(mesh.nodes[:, 0], mesh.nodes[:, 1], all_triangles,
                              vm / 1e6, levels=20, cmap="jet")
    plt.colorbar(contour, label="von Mises stress (MPa)")

    for row in mesh.connectivity:
        outline = []
        for index in range(len(boundary_order)):
            node = row[boundary_order[index]]
            outline.append(mesh.nodes[node])
        outline = np.asarray(outline)
        plt.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.5)

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Von Mises stress")
    plt.tight_layout()
    if save_path is not None :
        plt.savefig(save_path, dpi=200)
    plt.show()
    
    
    
