"""3-node constant strain triangle."""
import numpy as np 
from fesolver.elements.base import Base
from fesolver.quadrature import gauss_1d

class CST(Base) : 
    """
    Linear triangular element with constant strain and stress.

    Node ordering: 1, 2, 3 anticlockwise, at natural coordinates (0,0),
    (1,0) and (0,1) respectively.

    Shape functions are linear, so their derivatives are constant and B does
    not vary within the element. The stiffness matrix is therefore evaluated
    in closed form rather than by quadrature.
    """
    n_nodes = 3
    n_gauss = 1
    n_gauss_points = 1
    
    def shape_functions(self,xi,eta) : 
        """Shape function values at (xi, eta), shape (3,)."""
        N1 = 1 - xi - eta
        N2 = xi 
        N3 = eta
        N = np.array([N1,N2,N3])
        return N 
        
    def shape_derivatives(self, xi, eta):
        """
        Natural derivatives, shape (3, 2).

        Constant for this element, so xi and eta are accepted for interface
        consistency but unused.
        """
        dN1_dxi , dN1_deta = -1 , -1
        dN2_dxi , dN2_deta = 1 , 0
        dN3_dxi , dN3_deta = 0 , 1
        shape_derivatives = np.array([[dN1_dxi,dN1_deta], 
                                      [dN2_dxi,dN2_deta], 
                                      [dN3_dxi,dN3_deta]]) 
        return shape_derivatives
    
    def extrapolation_matrix(self) : 
        """Extrapolation operator, trivial for a constant-strain element.

        Stress is constant over a CST, so the single Gauss-point value applies
        unchanged at all three nodes. The general least-squares construction in
        the base class is singular here -- one known value cannot determine three
        independent nodal values -- so it is overridden with the known answer.

        Returns
        -------
        ndarray, shape (3, 1)
        """
        return np.ones((3,1))
    
    def stiffness(self, coords, material, thickness):
        """
        Element stiffness matrix, shape (6, 6).

        Closed form K = B.T @ D @ B * t * A, where the area A is half the
        Jacobian determinant. No quadrature is needed because the integrand
        is constant over the element.
        """
        B = self.B_matrix(0,0,coords)
        inv_jacobian, det_jacobian = self.jacobian(0,0,coords)
        K = B.T @ material.D @ B * thickness/2 * det_jacobian
        return K 
    
    def gauss_point_stresses(self, coords, material, u_e):
        """
        Strain and stress for the element.

        u_e is the element displacement vector, shape (6,), in DOF order
        [u1, v1, u2, v2, u3, v3].

        Returns strain and stress, each (1, 3) in Voigt ordering. Strain is
        constant over the element, so a single row is returned rather than
        one per quadrature point.
        """
        B = self.B_matrix(0,0,coords)
        strain = B @ u_e.flatten()
        stress = strain.T @ material.D.T
        return strain,stress

        
        
    
        

        
        
