"""Linear elastic material models. SI units throughout (m, N, Pa)."""
import numpy as np

class PlaneStressMaterial : 
    """
    Isotropic linear elastic material under plane stress.

    E : Young's modulus in Pa, must be positive.
    nu : Poisson's ratio, must satisfy -1 < nu < 0.5. The upper limit is
        excluded because nearly incompressible materials cause locking.

    D is the 3x3 constitutive matrix, sigma = D @ epsilon, using Voigt
    ordering [sigma_x, sigma_y, tau_xy] and [eps_x, eps_y, gamma_xy],
    where gamma_xy is the engineering shear strain.
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
        """Plane-stress constitutive matrix, recomputed from the current E and nu."""
        D = np.array([[1,self.nu,0],[self.nu,1,0],[0,0,(1-self.nu)/2]])
        D = self.E/(1-self.nu**2) * D
        return D
        

        
