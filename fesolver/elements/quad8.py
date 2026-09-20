import numpy as np
from fesolver.elements.base import Base
from fesolver.quadrature import gauss_2d

class Quad8(Base) : 
    
    """
    8-node quadrilateral, quadratic shape functions.

    Node ordering: corners 1-4 anticlockwise from (-1,-1), then midside
    nodes 5-8, where node 5 lies between nodes 1 and 2.

        node: 1        2       3      4       5       6       7       8
        (xi, eta): (-1,-1) (1,-1) (1,1) (-1,1) (0,-1) (1,0)  (0,1)  (-1,0)
    """
    
    n_nodes = 8 
    n_gauss = 3
    n_gauss_points = 9 
    
    def shape_functions(self,xi,eta) : 
        """Shape function values, shape (8,)."""
     
        N1 = -0.25*(1-xi)*(1-eta)*(1+xi+eta)
        N2 = -0.25*(1+xi)*(1-eta)*(1-xi+eta)
        N3 = -0.25*(1+xi)*(1+eta)*(1-xi-eta)
        N4 = -0.25*(1-xi)*(1+eta)*(1+xi-eta)
        N5 = 0.5*(1-xi**2)*(1-eta)
        N6 = 0.5*(1+xi)*(1-eta**2)
        N7 = 0.5*(1-xi**2)*(1+eta)
        N8 = 0.5*(1-xi)*(1-eta**2)
        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
        return N
    
    def shape_derivatives(self, xi, eta):
        """Natural derivatives, shape (8, 2): column 0 is dN/dxi, column 1 is dN/deta."""
        
        dN1_dxi , dN1_deta = 0.25 * (1 - eta) * (2 * xi + eta), 0.25 * (1 - xi) * (xi + 2 * eta)
        dN2_dxi , dN2_deta = 0.25 * (1 - eta) * (2 * xi - eta), 0.25 * (1 + xi) * (2 * eta - xi)
        dN3_dxi , dN3_deta = 0.25 * (1 + eta) * (2 * xi + eta), 0.25 * (1 + xi) * (xi + 2 * eta)
        dN4_dxi , dN4_deta = 0.25 * (1 + eta) * (2 * xi - eta), 0.25 * (1 - xi) * (2 * eta - xi)
        dN5_dxi , dN5_deta = -xi * (1 - eta),-0.5 * (1 - xi**2)
        dN6_dxi , dN6_deta = 0.5 * (1 - eta**2),-(1 + xi) * eta
        dN7_dxi , dN7_deta = -xi * (1 + eta),0.5 * (1 - xi**2)
        dN8_dxi , dN8_deta = -0.5 * (1 - eta**2),-(1 - xi) * eta
        shape_derivatives = np.array([[dN1_dxi,dN1_deta],
                                      [dN2_dxi,dN2_deta], 
                                      [dN3_dxi,dN3_deta],
                                      [dN4_dxi,dN4_deta],
                                      [dN5_dxi,dN5_deta],
                                      [dN6_dxi,dN6_deta],
                                      [dN7_dxi,dN7_deta],
                                      [dN8_dxi,dN8_deta]])
        return shape_derivatives
