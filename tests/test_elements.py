"""Tests for the element classes — shape functions, derivatives, stiffness
and stress recovery.

These are the only checks on the element mathematics, which is hand-derived:
shape function derivatives were written out analytically when SymPy was
dropped, so nothing else in the package verifies them.

Test points are interior and asymmetric. Corners and the element centre let
sign errors cancel, so a wrong shape function set can pass there.
"""

from fesolver.elements.quad8 import Quad8
from fesolver.elements.cst import CST
from fesolver.materials import PlaneStressMaterial
import numpy as np 
import pytest

POINTS = [(-0.73,  0.41),
          ( 0.12, -0.88),
          ( 0.64,  0.29),
          (-0.35, -0.17),
          ( 0.91,  0.76),
          (-0.05,  0.93)]
NODE_POINTS = [(-1, -1), (1, -1), (1, 1), (-1, 1),
               ( 0, -1), (1,  0), (0, 1), (-1, 0)]
LUG_COORDS = [[-0.0150, 0.000],
            [ 0.0000, 0.000],
            [ 0.0000, 0.100],
            [-0.0300, 0.060],
            [-0.0075, 0.000],
            [ 0.0000, 0.050],
            [-0.0150, 0.080],
            [-0.0225, 0.030]]
CST_COORDS = [[0,0], [0,0.1] , [-0.015,0.08]]
THICKNESS = 0.032

def test_shape_functions_partition_of_unity () : 
    quad8 = Quad8()
    for xi , eta  in POINTS : 
        sum = 0
        N = quad8.shape_functions(xi,eta)
        for i in N : 
            sum += i
        assert np.isclose(sum , 1)
   
def test_shape_functions_kronecker_delta () : 
    quad8 = Quad8()
    for  i , (x , y) in enumerate(NODE_POINTS): 
        N = quad8.shape_functions(x,y)
        N_popped = np.delete(N,i)
        assert N[i] == 1 and np.allclose(N_popped , 0)

def test_shape_derivatives_sum_to_zero () : 
    quad8 = Quad8()
    for x,y in NODE_POINTS : 
        sum_x = 0
        sum_y = 0 
        N = quad8.shape_derivatives(x,y)
        for col1 , col2 in N : 
            sum_x += col1
            sum_y += col2
        assert np.isclose(sum_x , 0) and np.isclose(sum_y , 0)
      
def test_shape_derivatives_match_finite_difference():
    """Analytical derivatives agree with a central difference of the shape
    functions.

    Covers all eight shape functions in both natural directions at six
    interior points. This is the check that would catch a transcription
    error in the hand-written derivatives.

    At h = 1e-5 the truncation error goes as h**2 (~1e-10) and the roundoff
    as eps/h (~1e-11), both well under the default absolute tolerance, which
    governs wherever a derivative is near zero.
    """
    h = 1e-5
    quad8 = Quad8()
    for xi , eta in POINTS : 
        N_perturbed1 = quad8.shape_functions(xi+h,eta)
        N_perturbed2 = quad8.shape_functions(xi-h,eta)
        N_perturbed3 = quad8.shape_functions(xi,eta+h)
        N_perturbed4 = quad8.shape_functions(xi,eta-h)
        dN = quad8.shape_derivatives(xi,eta)
        for index , (col1 , col2) in enumerate(dN) : 
            assert np.isclose((N_perturbed1[index] - N_perturbed2[index])/ (2*h) , col1 , rtol = 1e-6)
            assert np.isclose((N_perturbed3[index] - N_perturbed4[index])/ (2*h) , col2, rtol = 1e-6)

def test_quad8_extrapolation_rows_sum_to_one():
    """Extrapolation reproduces a constant stress field exactly.

    Each row holds the weights combining the nine Gauss-point stresses into
    one nodal value. If the weights did not sum to one, a uniform field
    would be scaled rather than reproduced -- an error of a few percent that
    would not be visible in a stress contour.
    """
    quad8 = Quad8()
    L = quad8.extrapolation_matrix()   
    assert np.allclose(L.sum(axis=1) ,1.0)
    
def test_cst_extrapolation_is_ones () : 
    cst = CST()
    L = cst.extrapolation_matrix()
    assert np.allclose(L , 1.0)

def test_quad8_stiffness_symmetric ( ) : 
    quad8 = Quad8()
    D = PlaneStressMaterial(209e9,0.3)
    K = quad8.stiffness(LUG_COORDS , D ,THICKNESS )
    assert np.allclose(K, K.T)
    
def test_quad8_stiffness_has_three_rigid_body_modes():
    """K is singular in exactly three directions: two translations and one
    rotation.

    The zero eigenvalues are only zero relative to the largest, so the
    tolerance is computed from the matrix rather than fixed -- stiffness
    magnitudes scale with modulus, thickness and element size.

    The second assertion, that the fourth eigenvalue is a meaningful
    fraction of the largest, is what distinguishes three rigid-body modes
    from a matrix that is simply empty.
    """
    quad8 = Quad8()
    D = PlaneStressMaterial(209e9,0.3)
    K = quad8.stiffness(LUG_COORDS , D ,THICKNESS )
    assert np.allclose(np.linalg.eigvalsh(K)[0:3] , 0 , 
                       atol = 1e-9 * abs(np.linalg.eigvalsh(K)[-1])) and np.linalg.eigvalsh(K)[3] / np.linalg.eigvalsh(K)[-1] > 1e-6

def test_cst_stiffness_has_three_rigid_body_modes () : 
    cst =CST()
    D = PlaneStressMaterial(209e9,0.3)
    K = cst.stiffness(CST_COORDS , D ,THICKNESS )
    assert np.allclose(np.linalg.eigvalsh(K)[0:3] , 0 , 
                       atol = 1e-9 * abs(np.linalg.eigvalsh(K)[-1])) and np.linalg.eigvalsh(K)[3] / np.linalg.eigvalsh(K)[-1]  > 1e-6
    
def test_rigid_translation_gives_zero_stress():
    """Translating every node by the same amount produces no stress.

    The residue is floating-point cancellation in B @ u, of order 1e-3 Pa
    against stresses of order 1e7 Pa in real use. The tolerance of 1 Pa sits
    three orders above that noise and seven below anything physically
    meaningful.
    """
    quad8 = Quad8()
    D = PlaneStressMaterial(209e9,0.3)
    strain_tensor , stress_tensor = quad8.gauss_point_stresses(LUG_COORDS,D,np.ones((16,)))
    assert np.allclose(stress_tensor,0 , atol=1)