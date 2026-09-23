"""Lifting lug under a vertical hook load — single 8-node element.

Reproduces a single-element benchmark: a steel lifting lug
analysed in plane stress, modelled with one 8-node serendipity element.

Problem
-------
Steel, E = 209 GPa, nu = 0.3, thickness 32 mm. Nodes 2, 3 and 6 are fully
fixed; a 58.86 kN vertical load is applied at node 7, representing a 6 tonne
hook load. Node numbering follows the original problem: corners 1-4 anticlockwise,
midsides 5-8.

Verification
------------
Displacements and nodal von Mises stresses are checked against values from
an earlier implementation, itself validated against an Abaqus model of
the same geometry. Peak von Mises is 95.61 MPa at node 3.

Note that the earlier script contained a Jacobian transpose error, since
corrected here; the uncorrected version gave a peak of 109 MPa.

Limitations
-----------
A single element is a coarse model and is used here because it is the case
with independently verified reference values. Mesh refinement and
discretisation error are addressed in the convergence study, not here.

Units are SI throughout (m, N, Pa); conversion to mm and MPa happens only in
the printed output.

Run from the repository root:

    python examples/lifting_lug.py
"""

import numpy as np
from fesolver.mesh import Mesh
from fesolver.materials import PlaneStressMaterial
from fesolver.elements.quad8 import Quad8
from fesolver.assembly import assembly_stiffness
from fesolver.solver import solve
from fesolver.postprocess import (element_gauss_stresses, element_nodal_stresses,
                                  average_nodal_stresses, von_mises)

u_ref = np.array([[ 5.03059646e-09,  1.34913364e-06],
                  [ 0.00000000e+00,  0.00000000e+00],
                  [ 0.00000000e+00,  0.00000000e+00],
                  [ 1.31616069e-06,  6.62628356e-06],
                  [-1.52292337e-07, -1.52573064e-07],
                  [ 0.00000000e+00,  0.00000000e+00],
                  [ 8.17320742e-07,  6.82205000e-06],
                  [-9.76877543e-08,  1.48733477e-06]])

vm_ref = np.array([43.93350567e6, 20.67163553e6, 95.61033722e6, 51.30850546e6,
                   11.23801127e6, 42.93049745e6, 57.34655092e6, 24.59230153e6])

if __name__ == "__main__" : 
    material = PlaneStressMaterial(209e9, 0.3)
nodes = [[-0.0150, 0.000],
         [ 0.0000, 0.000],
         [ 0.0000, 0.100],
         [-0.0300, 0.060],
         [-0.0075, 0.000],
         [ 0.0000, 0.050],
         [-0.0150, 0.080],
         [-0.0225, 0.030]]
mesh = Mesh(nodes, [[0, 1, 2, 3, 4, 5, 6, 7]])
t = 32/1000
K = assembly_stiffness(mesh, Quad8, material, t)
F = np.zeros(mesh.n_dofs)
F[mesh.node_dofs([6])[1]] = 58860 # Load in N 
u = solve(K, F, mesh.node_dofs([1, 2, 5]))

strain, stress = element_gauss_stresses(mesh, u, Quad8, material)
nodal = element_nodal_stresses(mesh, stress, Quad8)
avg = average_nodal_stresses(mesh, nodal)
vm = von_mises(avg)

print('--'*70)
print("Displacements match reference:", np.allclose(u.reshape(8,2), u_ref , atol = 0))
print("von Mises matches reference:   ", np.allclose(vm, vm_ref))
print(f"Peak von Mises: {vm.max()/1e6:.2f} MPa at node {vm.argmax() + 1}")

print('--'*70)
print('Displacement (mm)')
u = u.reshape(8,2)
for i, (x,y) in enumerate(u) : 
    print(f' Node {i+1}:  u = {x*1000:.6f} mm, v = {y*1000:.6f} mm ')
    
print('--'*70)
print('Average Nodal Stresses')
for i,stress_row in enumerate(avg) : 
    sx, sy, txy = stress_row / 1e6
    print(f' Node {i+1} Stress = [{sx:7.2f} {sy:7.2f} {txy:7.2f}] MPa')
    
print('--'*70)
print('von Mises Stress')
for i,vm_node in enumerate(vm) : 
    print(f' Node {i+1} von Mises Stress = {vm_node/1000000:.2f} MPa')


