from fesolver.mesh import Mesh 
from fesolver.materials import PlaneStressMaterial
from fesolver.elements.cst import CST 
from fesolver.elements.quad8 import Quad8 
import numpy as np 

def assembly_stiffnes(mesh,element_cls,material,thickness) : 
    K = np.zeros((mesh.n_dof,mesh.n_dof))
    for element in mesh.connectivity : 
        coords = mesh.element_coords(element)
        element_cls(coords,material,thickness)
        k_e = element.stiffness()
        dofs = mesh.element.dofs(element)
        K[np.ix_(dofs,dofs)] += k_e
    return K 



