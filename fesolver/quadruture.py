import numpy as np 

def gauss_1d(n:int) :  
    points , weights = np.polynomial.legendre.leggauss(n)
    return points , weights
        
def gauss_2d(n:int) : 
    x , w = np.polynomial.legendre.leggauss(n)
    points = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T

    print(points)
    
    
gauss_2d(2)