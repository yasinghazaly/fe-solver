"""Convergence study: end-loaded cantilever against its exact solution.

A rectangular beam is held along its left end and loaded by a parabolic
shear traction on its right end. This problem has a closed-form elasticity
solution whose displacement is cubic, so quadratic elements cannot
reproduce it exactly and the error can be measured as the mesh is refined.

The left-end nodes are held at their exact displacements, which are small
but not zero, and the right end carries the exact shear distribution, so
the finite element problem and the exact solution describe the same beam.

Theory predicts that for 8-node quadratic elements the L2 displacement
error falls as h**3 and the energy error as h**2. The script prints the
measured rates and saves a log-log plot.

Units are SI throughout. Run from the repository root:

    python examples/cantilever_convergence.py
"""

import numpy as np
import matplotlib.pyplot as plt

from fesolver.mesh import structured_quad8
from fesolver.elements.quad8 import Quad8
from fesolver.materials import PlaneStressMaterial
from fesolver.assembly import assembly_stiffness
from fesolver.loads import edge_traction
from fesolver.solver import solve
from fesolver.errors import l2_error, energy_error

LENGTH = 2.0         
HEIGHT = 0.5          
THICKNESS = 0.01      
LOAD = 1000.0         
E = 209e9            
NU = 0.3
I = THICKNESS * HEIGHT**3 / 12     
STEEL = PlaneStressMaterial(E, NU)

GRIDS = [(4, 1), (8, 2), (16, 4), (32, 8)]     


def exact_displacement(x, y):
    """Exact displacement (ux, uy) in metres at position (x, y)."""
    yc = y - HEIGHT / 2             
    ux = -LOAD * yc / (6 * E * I) * (
        (6 * LENGTH - 3 * x) * x + (2 + NU) * (yc**2 - HEIGHT**2 / 4)
    )
    uy = LOAD / (6 * E * I) * (
        3 * NU * yc**2 * (LENGTH - x)
        + (4 + 5 * NU) * HEIGHT**2 * x / 4
        + (3 * LENGTH - x) * x**2
    )
    return (ux, uy)


def exact_strain(x, y):
    """Exact strain (eps_x, eps_y, gamma_xy) at position (x, y)."""
    yc = y - HEIGHT / 2
    eps_x = -LOAD * yc * (LENGTH - x) / (E * I)
    eps_y = NU * LOAD * yc * (LENGTH - x) / (E * I)
    gamma_xy = LOAD * (1 + NU) / (E * I) * (HEIGHT**2 / 4 - yc**2)
    return (eps_x, eps_y, gamma_xy)


def end_traction(x, y):
    """Traction (tx, ty) in pascals on the right end: a parabolic shear."""
    yc = y - HEIGHT / 2
    return (0.0, LOAD / (2 * I) * (HEIGHT**2 / 4 - yc**2))


def solve_cantilever(nx, ny):
    """Solve the beam on an nx by ny grid and measure both errors.

    Returns the element size, the L2 displacement error and the energy
    error.
    """
    mesh = structured_quad8(LENGTH, HEIGHT, nx, ny)
    K = assembly_stiffness(mesh, Quad8, STEEL, THICKNESS)

    F = np.zeros(mesh.n_dofs)
    for e in range(mesh.n_elements):
        row = mesh.connectivity[e]
        edge_nodes = [row[1], row[5], row[2]]
        if np.allclose(mesh.nodes[edge_nodes][:, 0], LENGTH):
            F += edge_traction(mesh, edge_nodes, end_traction, THICKNESS)

    fixed_dofs = []
    fixed_values = []
    for node, (x, y) in enumerate(mesh.nodes):
        if np.isclose(x, 0.0):
            ux, uy = exact_displacement(x, y)
            fixed_dofs += [2 * node, 2 * node + 1]
            fixed_values += [ux, uy]

    u = solve(K, F, fixed_dofs, fixed_values)

    h = LENGTH / nx
    displacement_error = l2_error(mesh, u, Quad8, exact_displacement)
    strain_error = energy_error(mesh, u, Quad8, STEEL, exact_strain)
    return h, displacement_error, strain_error


def convergence_rate(h, error):
    """Rate between each pair of consecutive meshes: log(error ratio) / log(size ratio)."""
    h = np.asarray(h)
    error = np.asarray(error)
    return np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])


if __name__ == "__main__":
    sizes, l2_errors, energy_errors = [], [], []
    for nx, ny in GRIDS:
        h, displacement_error, strain_error = solve_cantilever(nx, ny)
        sizes.append(h)
        l2_errors.append(displacement_error)
        energy_errors.append(strain_error)

    l2_rates = convergence_rate(sizes, l2_errors)
    energy_rates = convergence_rate(sizes, energy_errors)

    print(f"{'grid':>8} {'h (m)':>9} {'L2 error':>12} {'rate':>6} {'energy error':>14} {'rate':>6}")
    for i, (nx, ny) in enumerate(GRIDS):
        l2_rate = f"{l2_rates[i - 1]:.2f}" if i > 0 else ""
        energy_rate = f"{energy_rates[i - 1]:.2f}" if i > 0 else ""
        print(f"{nx:>4} x {ny:<2} {sizes[i]:>9.4f} {l2_errors[i]:>12.4e} {l2_rate:>6} "
              f"{energy_errors[i]:>14.4e} {energy_rate:>6}")
    print()
    print(f"Final L2 rate:     {l2_rates[-1]:.2f}   (theory 3)")
    print(f"Final energy rate: {energy_rates[-1]:.2f}   (theory 2)")

    sizes = np.array(sizes)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(sizes, l2_errors, "o-", label=f"L2 error, rate {l2_rates[-1]:.2f}")
    ax.loglog(sizes, energy_errors, "s-", label=f"Energy error, rate {energy_rates[-1]:.2f}")
    ax.loglog(sizes, l2_errors[0] * (sizes / sizes[0]) ** 3, "k--", linewidth=0.8, label="slope 3")
    ax.loglog(sizes, energy_errors[0] * (sizes / sizes[0]) ** 2, "k:", linewidth=0.8, label="slope 2")
    ax.set_xlabel("element size h (m)")
    ax.set_ylabel("error")
    ax.set_title("Cantilever convergence, 8-node quadrilateral")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()