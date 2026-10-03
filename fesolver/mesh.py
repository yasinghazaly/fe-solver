import numpy as np 

class Mesh : 
    """
    Nodes and element connectivity for a 2D mesh.

    Pure geometry and topology: no material, loads, boundary conditions or
    element type. Each node carries two DOFs (u, v), so global node i owns
    DOFs 2i and 2i+1.

    Parameters
    ----------
    nodes : array_like, shape (n_nodes, 2)
        Nodal coordinates in metres, one node per row.
    connectivity : array_like, shape (n_elements, nodes_per_element)
        Global node indices for each element, listed in the element's local
        node order as defined by its shape functions.
    """
    def __init__(self , nodes:list , connectivity:list) : 
        self.nodes = np.asarray(nodes,dtype=float)
        self.connectivity = np.asarray(connectivity,dtype=int)
        
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2 : 
            raise ValueError(f'Nodes are the wrong shape') 
        if self.connectivity.ndim != 2 :
            raise ValueError(f'Connectivity is the wrong shape')

        self.n_nodes = len(self.nodes)
        self.n_elements = self.connectivity.shape[0]
        self.nodes_per_element = self.connectivity.shape[1]
        self.n_dofs = self.n_nodes * 2
        
        for index , element in enumerate(self.connectivity) : 
            if element.max() >= self.n_nodes : 
                raise IndexError(f'Index {element.max()} of Element {element} is out of range')
            if element.min() < 0 :
                raise IndexError(f'Index {element.min()} of Element {element} is negative')
            if len(np.unique(element)) != self.nodes_per_element : 
                raise ValueError(f'Element {index} repeats nodes {element}')
        
        used = np.unique(self.connectivity)
        if len(used) != self.n_nodes:
            orphans = np.setdiff1d(np.arange(self.n_nodes), used)
            raise ValueError(f'Nodes {orphans} are not used by any element')

    def element_coords(self,e:int) : 
        """
        Nodal coordinates of element `e`.

        Parameters
        ----------
        e : int
            Element index.

        Returns
        -------
        ndarray, shape (nodes_per_element, 2)
            Coordinates in the element's local node order.
        """
        global_nodes = self.connectivity[e]
        coords = []
        for i in global_nodes : 
            coord = self.nodes[i]
            coords.append(coord)
        return coords

    def element_dofs(self,e:int) : 
        """Global DOF indices for element `e`.

        Ordered to match the rows and columns of the element stiffness
        matrix: local node k contributes DOFs 2i and 2i+1 at positions 2k
        and 2k+1, where i is the global index of local node k. Never sorted
        -- connectivity order is what makes the mapping correct.

        Parameters
        ----------
        e : int
            Element index.

        Returns
        -------
        ndarray, shape (2 * nodes_per_element,)
        """
        global_nodes = self.connectivity[e]
        dofs = self.node_dofs(global_nodes)
        return dofs

    def node_dofs(self,nodes:list) : 
        """
        Global DOF indices for a set of nodes.

        Used to express boundary conditions and point loads in terms of node
        numbers rather than DOF numbers. Caller order is preserved.

        Parameters
        ----------
        nodes : array_like of int
            Global node indices.

        Returns
        -------
        ndarray, shape (2 * len(nodes),)
        """
        dofs = []
        for i in nodes : 
            node = 2*i
            dofs.append(node)
            dofs.append(node+1)
        return dofs

    def check_orientation(self , n_corners = 3) : 
        """
        Find elements whose nodes are not ordered counterclockwise.

        Clockwise ordering gives a negative Jacobian determinant and a
        sign-reversed stiffness matrix, which the solve will not flag. The
        signed area is computed by the shoelace formula over the first
        `n_corners` entries of each connectivity row; midside nodes do not
        affect orientation and are excluded.

        Parameters
        ----------
        n_corners : int, optional
            Number of leading corner nodes per element: 3 for CST, 4 for
            Quad8. Default 3.

        Returns
        -------
        list of int
            Indices of elements with non-positive signed area. Empty if all
            elements are correctly oriented.
        """
        anticlockwise_ordering = [] 
        for i in range(self.n_elements) : 
            coords = self.element_coords(i)   
            A = 0 
            for j in range(n_corners) : 
                A += (coords[j][0] * coords[(j+1)%n_corners][1]) - (coords[(j+1)%n_corners][0] * coords[j][1])
            if A <= 1e-12 : 
                anticlockwise_ordering.append(i)
        return anticlockwise_ordering
            

def structured_quad8(length, height, nx, ny ) : 
    """Generate a rectangular grid of 8-node quadrilateral elements.

    Nodes lie on a lattice with twice the element resolution: 2*nx + 1
    columns and 2*ny + 1 rows. Lattice points where both the column and
    row index are odd are element centres, which a serendipity element
    does not have, so they are skipped. The remaining points are numbered
    row by row from the bottom-left.

    Each element's connectivity is then read from a lookup table of node
    numbers in the Quad8 local order: corners anticlockwise from
    bottom-left, followed by midside nodes starting from the bottom edge.

    Parameters
    ----------
    length : float
        Extent in x, in metres.
    height : float
        Extent in y, in metres.
    nx : int
        Number of elements along x.
    ny : int
        Number of elements along y.

    Returns
    -------
    Mesh
        With (2*nx + 1)*(2*ny + 1) - nx*ny nodes and nx*ny elements,
        numbered row by row. The bottom-left corner is at the origin.
    """
    n_cols = (2 * nx) + 1 
    n_rows = (2 * ny) + 1
    dx = length / (2 * nx)   
    dy = height / (2 * ny)   
    node_coords = []
    counter = 0 
    lookup_table = np.ones((n_rows,n_cols) , dtype = int) * -1
    connectivity = []
    
    for row in range(n_rows) : 
        for col in range(n_cols)  :
            if row * col % 2 == 0 : 
                lookup_table[row,col] = counter
                node_coords.append((col*dx,row*dy))
                counter += 1 
      
    for ey in range(ny) : 
        for ex in range(nx) : 
            col = 2 *ex
            row = 2* ey 
            connectivity.append([
                lookup_table[row,col],
                lookup_table[row,col+2],
                lookup_table[row+2,col+2],
                lookup_table[row+2,col],
                lookup_table[row,col+1],
                lookup_table[row+1,col+2],
                lookup_table[row+2,col+1],
                lookup_table[row+1,col],
            ])
    return Mesh(node_coords , connectivity)






