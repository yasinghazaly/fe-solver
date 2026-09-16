"""
Linear elastic material models for 2D finite element analysis.

Units are SI throughout the package: lengths in m, forces in N,
stresses and moduli in Pa.

"""
import numpy as np

class PlaneStressMaterial : 
    """
    Isotropic linear elastic material under plane stress.

    Parameters
    ----------
    E : float
        Young's modulus in Pa. Must be positive.
    nu : float
        Poisson's ratio. Must satisfy -1 < nu < 0.5.

    Attributes
    ----------
    E : float
        Young's modulus in Pa.
    nu : float
        Poisson's ratio.
    D : ndarray, shape (3, 3)
        Plane-stress constitutive matrix relating stress to strain,
        sigma = D @ epsilon.

    Raises
    ------
    ValueError
        If E <= 0 or nu is outside (-1, 0.5).

    Notes
    -----
    Voigt notation is used throughout the package:

    - stress: [sigma_x, sigma_y, tau_xy]
    - strain: [eps_x, eps_y, gamma_xy]

    gamma_xy is the engineering shear strain (2 * eps_xy), which is why
    the shear term of D is (1 - nu) / 2:

        D = E / (1 - nu**2) * [[1,  nu, 0           ],
                               [nu, 1,  0           ],
                               [0,  0,  (1 - nu) / 2]]

    Plane stress assumes the out-of-plane stresses are zero, which suits
    bodies that are thin compared with their in-plane dimensions.
    nu = 0.5 is excluded because nearly incompressible materials cause
    locking in standard displacement-based elements.
    """
        
    def __init__(self,E,nu) :
        
        if E <= 0 : 
            raise ValueError("Young's Modulus must be positive.")
        self.E = E
        
        if nu >= 0.5 or nu <= -1 : 
            raise ValueError("Poisson's ratio must lie bwteen -1.0 and 0.5 ")
        self.nu = nu
            
    @property
    def D(self) : 
        D = np.array([[1,self.nu,0],[self.nu,1,0],[0,0,(1-self.nu)/2]])
        D = self.E/(1-self.nu**2) * D
        return D
        

        
