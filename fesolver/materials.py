import numpy as np

class plane_stress_constitutive_material_matrix : 
    def __init__(self,E,nu) :
        
        if E <= 0 : 
            raise ValueError("young's Modulus must be a positive integer.")
        else : 
            self.E = E
        
        if nu > 0.5 or nu < -1 : 
            raise ValueError("Poisson's ratio must lie bwteen -1.0 and 0.5 ")
        else : 
            self.nu = nu
            
        self.D = self.construct_matrix()

    def construct_matrix (self) : 
        D = np.array([[1,self.nu,0],[self.nu,1,0],[0,0,(1-self.nu)/2]])
        D = self.E/(1-self.nu**2) * D
        
        return D
        
        
        
