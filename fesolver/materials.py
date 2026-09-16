import numpy as np

class plane_stress_constitutive_material_matrix : 
    def __init__(self,E,nu) :
        self.E = E
        self.nu = nu
        self.construct_matrix()

    def construct_matrix (self,) : 
        D = np.array([1,self.nu,0],[self.nu,1,0],[0,0,(1-self.nu)/2])
        D = self.E/(1-self.nu**2) @ D
        
        return D
        
        
        
