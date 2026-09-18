"""Base class for 2D isoparametric elements. SI units throughout (m, N, Pa)."""
from abc import ABC, abstractmethod
import numpy as np
from fesolver.quadrature import gauss_2d


class Base(ABC) : 
    """
    Shared machinery for 2D isoparametric elements.

    Subclasses provide shape_functions and shape_derivatives, plus the class
    attributes n_nodes and n_gauss. Everything else is inherited.

    Conventions:
    - coords : (n_nodes, 2) array of nodal positions in m, one node per row.
    - element DOF order : [u1, v1, u2, v2, ...], which fixes the columns of B.
    - natural derivatives : (n_nodes, 2), column 0 is dN/dxi, column 1 is dN/deta.
    - Voigt ordering follows fesolver.materials.
    """
    @abstractmethod
    def shape_functions(self,xi,eta) :
        """Shape function values at (xi, eta), shape (n_nodes,)."""
        ...
    
    @abstractmethod
    def shape_derivatives(self,xi,eta) :
        """Natural derivatives at (xi, eta), shape (n_nodes, 2)."""
        ...
        
    def jacobian(self,xi,eta,coords) : 
        """
        Jacobian of the mapping from natural to real coordinates.

        J has xi and eta as rows, x and y as columns, so the real derivatives
        are obtained from inv(J) with no transpose.

        Returns inv(J), shape (2, 2), and det(J).
        """
        J = np.zeros((2,2))
        natural_derivatives = self.shape_derivatives(xi,eta)
        for i in range(len(natural_derivatives)) : 
            J[0,0] += natural_derivatives[i][0] * coords[i][0]
            J[0,1] += natural_derivatives[i][0] * coords[i][1]
            J[1,0] += natural_derivatives[i][1] * coords[i][0]
            J[1,1] += natural_derivatives[i][1] * coords[i][1]
        inverse_jacobian = np.linalg.inv(J)
        det_jacobian = np.linalg.det(J)
        return inverse_jacobian , det_jacobian
    
    def real_derivatives(self,xi,eta,coords) : 
        """Derivatives with respect to x and y, shape (n_nodes, 2)."""
        real_derivatives = []
        natural_derivatives = self.shape_derivatives(xi,eta)
        inverse_jacobian , det_jacobian = self.jacobian(xi,eta,coords)
        for i in range(len(natural_derivatives)) : 
            real_derivatives.append(inverse_jacobian @ natural_derivatives[i])
        real_derivatives = np.array(real_derivatives)
        return real_derivatives
        
    def B_matrix(self,xi,eta,coords) : 
        """Strain-displacement matrix at (xi, eta), shape (3, 2 * n_nodes)."""
        real_derivatives = self.real_derivatives(xi,eta,coords)
        B = np.zeros((3,2*self.n_nodes))
        for i in range(len(real_derivatives)) :
            B[0,2*i] = real_derivatives[i][0] 
            B[1,2*i + 1] = real_derivatives[i][1] 
            B[2,2*i] = real_derivatives[i][1] 
            B[2,2*i + 1] = real_derivatives[i][0] 
        return B
    
    def stiffness(self,coords,material,thickness) : 
        """
        Element stiffness matrix by Gauss quadrature, shape (2*n_nodes, 2*n_nodes).

        thickness is in m. Raises ValueError if det(J) <= 0 at any Gauss point,
        which means the element is degenerate or its nodes are ordered clockwise.
        """
        points , weights = gauss_2d(self.n_gauss) 
        D = material.D
        K = np.zeros((2*self.n_nodes,2*self.n_nodes))
        for k ,(xi,eta) in enumerate(points) : 
            inv_jacobian, det_jacobian = self.jacobian(xi,eta,coords)
            if det_jacobian <= 0 : 
                    raise ValueError("Jacobian must be greater than 0")
            else : 
                B = self.B_matrix(xi,eta,coords)
                K += B.T @  D @ B *det_jacobian *weights[k] * thickness
                    
        return K
    
    def gauss_point_stresses(self,coords,material,u_e) : 
        """
        Strain and stress at each Gauss point.

        u_e is the element displacement vector, shape (2 * n_nodes,), in DOF
        order [u1, v1, u2, v2, ...].

        Returns strain and stress, each (n_gauss**2, 3) in Voigt ordering,
        in the same point order as quadrature.gauss_2d.
        """
        points, weights = gauss_2d(self.n_gauss)
        strain = np.zeros((len(points),3))
        for k , (xi,eta) in enumerate(points) : 
            B = self.B_matrix(xi,eta,coords)
            strain[k] = B @ u_e.flatten()
        stress = strain @ material.D.T
        return strain , stress
            


    
        
        
    