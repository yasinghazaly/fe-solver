"""Benchmark of the sparse solvers against dense and SciPy direct solves.

An end-loaded cantilever is solved on four Quad8 grids, each with four
times the degrees of freedom of the one before. On every grid the reduced
system K_ff u = F_rhs is solved by

- a dense direct solve (NumPy), skipped once K_ff is too large to store densely,
- a sparse direct solve (SciPy spsolve), used as the reference answer,
- conjugate gradients with no preconditioner, Jacobi and IC(0).

Solve time, CG iterations, storage of K_ff and the error of each answer
against the reference are recorded. A second pass times each stage of the
IC(0) path on its own, and checks that the stages add up to the measured
IC(0) solve time.

A table is printed and three log-log plots are saved to docs/images/.
The largest grid takes a few minutes; run from any directory with

    python examples/benchmark.py
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve

from fesolver.assembly import assemble_triplets
from fesolver.elements.quad8 import Quad8
from fesolver.iterative import cg
from fesolver.materials import PlaneStressMaterial
from fesolver.mesh import structured_quad8
from fesolver.solver import partition_csr
from fesolver.sparse import CSR

GRIDS = [(16, 4), (32, 8), (64, 16), (128, 32)]
LENGTH, HEIGHT, THICKNESS = 2.0, 0.5, 0.01
STEEL = PlaneStressMaterial(209e9, 0.3)
LOAD = -1000.0
DENSE_LIMIT = 10_000   # largest K_ff solved densely (10,000 DOFs is 0.8 GB)
REPEATS = 10           # calls averaged when timing one matvec or one solve
IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"


def loads_and_supports(mesh):
    """Load vector and fixed DOFs for the cantilever.

    Both DOFs of every node on the left end (x = 0) are held at zero, and
    a point load of LOAD newtons in y acts at the top-right corner.

    Parameters
    ----------
    mesh : Mesh

    Returns
    -------
    F : ndarray, shape (n_dofs,)
        Global load vector.
    fixed_dofs : list of int
        Global indices of the constrained DOFs.
    """
    F = np.zeros(mesh.n_dofs)
    corner = [i for i, (x, y) in enumerate(mesh.nodes)
              if np.isclose(x, LENGTH) and np.isclose(y, HEIGHT)][0]
    F[2 * corner + 1] = LOAD
    left_nodes = [i for i, (x, _) in enumerate(mesh.nodes) if np.isclose(x, 0.0)]
    fixed_dofs = [d for node in left_nodes for d in (2 * node, 2 * node + 1)]
    return F, fixed_dofs


def timed(function, *args):
    """Call function(*args) once; return its result and the time taken in seconds."""
    start = time.perf_counter()
    output = function(*args)
    return output, time.perf_counter() - start


def relative_error(u, u_reference):
    """Relative error |u - u_reference| / |u_reference| in the 2-norm."""
    return np.linalg.norm(u - u_reference) / np.linalg.norm(u_reference)


def compare_solvers(nx, ny):
    """Solve one grid with every solver and record time, error and storage.

    Parameters
    ----------
    nx, ny : int
        Number of elements along x and y.

    Returns
    -------
    dict
        Grid name, total and free DOFs; for each solver its time, and for
        dense and the three CG runs their error against SciPy; CG iteration
        counts; storage of K_ff in MB, dense and CSR. Dense time and error
        are None above DENSE_LIMIT free DOFs, where the dense solve is
        skipped. Dense storage is always computed.
    """
    mesh = structured_quad8(LENGTH, HEIGHT, nx, ny)
    rows, cols, vals = assemble_triplets(mesh, Quad8, STEEL, THICKNESS)
    K = CSR(rows, cols, vals, mesh.n_dofs)
    F, fixed_dofs = loads_and_supports(mesh)
    K_ff, F_rhs, _, _ = partition_csr(K, F, fixed_dofs)
    n = K_ff.n

    result = {
        'Grid': f'{nx}x{ny}',
        'DOFs': mesh.n_dofs,
        'Free DOFs': n,
        'Dense Memory': n * n * 8 / 1e6,
        'CSR Memory': (K_ff.data.nbytes + K_ff.indices.nbytes + K_ff.indptr.nbytes) / 1e6,
    }

    # Sparse direct solve: the reference answer. The SciPy matrix shares
    # the three CSR arrays, so building it is not timed.
    K_scipy = csr_array((K_ff.data, K_ff.indices, K_ff.indptr), shape=(n, n))
    u_reference, result['SciPy Time'] = timed(spsolve, K_scipy, F_rhs)

    # Dense direct solve. to_dense() is not timed.
    if n < DENSE_LIMIT:
        K_dense = K_ff.to_dense()
        u_dense, result['Dense Time'] = timed(np.linalg.solve, K_dense, F_rhs)
        result['Dense Error'] = relative_error(u_dense, u_reference)
    else:
        result['Dense Time'] = None
        result['Dense Error'] = None

    for name, preconditioner in [('No Preconditioner', None),
                                 ('Jacobi', 'jacobi'),
                                 ('IC(0)', 'ic0')]:
        (u, history), result[f'{name} Time'] = timed(
            lambda: cg(K_ff, F_rhs, preconditioner=preconditioner))
        result[f'{name} Iterations'] = len(history) - 1
        result[f'{name} Error'] = relative_error(u, u_reference)

    return result


def stage_breakdown(nx, ny):
    """Time each stage of the IC(0) path on one grid.

    Stages that run once per solve (element loop, CSR build, partitioning,
    IC(0) factorisation, transpose) are timed once. Stages that run once
    per CG iteration (one matvec, one forward plus backward solve) are
    averaged over REPEATS calls, because a single call is too short to time
    reliably. The input vector is all ones: the cost depends on the
    sparsity pattern, not on the values.

    Parameters
    ----------
    nx, ny : int
        Number of elements along x and y.

    Returns
    -------
    dict
        Grid name and the time of each stage in seconds.
    """
    mesh = structured_quad8(LENGTH, HEIGHT, nx, ny)
    F, fixed_dofs = loads_and_supports(mesh)

    stages = {'Grid': f'{nx}x{ny}'}
    (rows, cols, vals), stages['Element Loop'] = timed(
        assemble_triplets, mesh, Quad8, STEEL, THICKNESS)
    K, stages['CSR Build'] = timed(CSR, rows, cols, vals, mesh.n_dofs)
    (K_ff, _, _, _), stages['Partitioning'] = timed(partition_csr, K, F, fixed_dofs)
    L, stages['IC(0) Factorisation'] = timed(K_ff.incomplete_cholesky)
    L_transposed, stages['Transpose'] = timed(L.transpose)

    x = np.ones(K_ff.n)
    start = time.perf_counter()
    for _ in range(REPEATS):
        K_ff.matvec(x)
    stages['Matvec'] = (time.perf_counter() - start) / REPEATS

    start = time.perf_counter()
    for _ in range(REPEATS):
        L.backward_solve(L_transposed, L.forward_solve(L, x))
    stages['Solve'] = (time.perf_counter() - start) / REPEATS

    return stages


def check_breakdown(stages, result, rtol=0.2):
    """Check that the timed stages add up to the measured IC(0) solve time.

    The predicted time is the one-off setup (factorisation and transpose)
    plus the number of iterations times the cost of one iteration (one
    matvec and one forward plus backward solve). The vector updates and
    dot products inside CG are not timed, so the prediction runs slightly
    low on small grids.

    Parameters
    ----------
    stages : dict
        Output of stage_breakdown for one grid. Updated in place with
        'Predicted', 'Ratio' (predicted / measured) and 'Check' ('ok' if
        within rtol of the measured time, otherwise 're-run').
    result : dict
        Output of compare_solvers for the same grid.
    rtol : float, optional
        Relative tolerance. Defaults to 0.2.
    """
    measured = result['IC(0) Time']
    predicted = (stages['IC(0) Factorisation'] + stages['Transpose']
                 + result['IC(0) Iterations'] * (stages['Matvec'] + stages['Solve']))
    stages['Predicted'] = predicted
    stages['Ratio'] = predicted / measured
    stages['Check'] = 'ok' if np.isclose(measured, predicted, rtol=rtol) else 're-run'


def print_tables(results, breakdowns):
    """Print the solver comparison and the stage breakdown, one row per grid."""
    def fmt(value):
        return '-' if value is None else f'{value:.3g}'

    print('\nSolve time (s), CG iterations in brackets, storage of K_ff (MB)')
    print(f"{'Grid':>7} {'Free DOFs':>9} {'Dense':>8} {'SciPy':>8} "
          f"{'CG':>13} {'Jacobi':>13} {'IC(0)':>13} {'Dense MB':>9} {'CSR MB':>7}")
    for r in results:
        cg_cols = [f"{fmt(r[f'{k} Time'])} ({r[f'{k} Iterations']})"
                   for k in ('No Preconditioner', 'Jacobi', 'IC(0)')]
        print(f"{r['Grid']:>7} {r['Free DOFs']:>9} {fmt(r['Dense Time']):>8} "
              f"{fmt(r['SciPy Time']):>8} {cg_cols[0]:>13} {cg_cols[1]:>13} "
              f"{cg_cols[2]:>13} {fmt(r['Dense Memory']):>9} {fmt(r['CSR Memory']):>7}")

    worst = max(r[f'{k} Error'] for r in results
                for k in ('No Preconditioner', 'Jacobi', 'IC(0)', 'Dense')
                if r[f'{k} Error'] is not None)
    print(f'Largest relative error against SciPy spsolve: {worst:.1e}')

    keys = ['Element Loop', 'CSR Build', 'Partitioning', 'IC(0) Factorisation',
            'Transpose', 'Matvec', 'Solve']
    print('\nStage breakdown (s); Matvec and Solve are per call')
    print(f"{'Grid':>7} " + ' '.join(f'{k:>{max(len(k), 8)}}' for k in keys)
          + f" {'Predicted':>9} {'Ratio':>6} {'Check':>6}")
    for s in breakdowns:
        print(f"{s['Grid']:>7} " + ' '.join(f'{fmt(s[k]):>{max(len(k), 8)}}' for k in keys)
              + f" {fmt(s['Predicted']):>9} {s['Ratio']:>6.2f} {s['Check']:>6}")


def plot_lines(x, series, ylabel, title, filename):
    """Save a log-log plot with one line per series.

    Parameters
    ----------
    x : list of int
        Free DOFs of each grid.
    series : list of (str, list)
        Label and values for each line. None values are left out, so a
        line can stop early (dense beyond DENSE_LIMIT).
    ylabel, title, filename : str
    """
    plt.figure(figsize=(7, 5))
    for label, values in series:
        points = [(xi, yi) for xi, yi in zip(x, values) if yi is not None]
        plt.loglog(*zip(*points), marker='o', label=label)
    plt.xlabel('Free DOFs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / filename, dpi=200)


def plot_results(results):
    """Save the time, storage and iteration plots to docs/images/."""
    x = [r['Free DOFs'] for r in results]

    def column(key):
        return [r[key] for r in results]

    plot_lines(x, [('Dense direct (NumPy)', column('Dense Time')),
                   ('CG, no preconditioner', column('No Preconditioner Time')),
                   ('CG, Jacobi', column('Jacobi Time')),
                   ('CG, IC(0)', column('IC(0) Time')),
                   ('Sparse direct (SciPy spsolve)', column('SciPy Time'))],
               'Solve time (s)', 'Solve time against problem size',
               'benchmark_time.png')
    plot_lines(x, [('Dense', column('Dense Memory')),
                   ('CSR', column('CSR Memory'))],
               'Storage of K_ff (MB)', 'Matrix storage against problem size',
               'benchmark_memory.png')
    plot_lines(x, [('No preconditioner', column('No Preconditioner Iterations')),
                   ('Jacobi', column('Jacobi Iterations')),
                   ('IC(0)', column('IC(0) Iterations'))],
               'CG iterations', 'CG iterations against problem size',
               'benchmark_iterations.png')


def main(grids=GRIDS):
    """Run the benchmark on every grid, print the tables and save the plots."""
    # The first spsolve call pays a one-off set-up cost; pay it here, untimed.
    spsolve(csr_array(np.eye(2)), np.ones(2))

    results, breakdowns = [], []
    for nx, ny in grids:
        print(f'Running {nx}x{ny}...', flush=True)
        result = compare_solvers(nx, ny)
        stages = stage_breakdown(nx, ny)
        check_breakdown(stages, result)
        results.append(result)
        breakdowns.append(stages)

    print_tables(results, breakdowns)
    plot_results(results)
    plt.show()
    return results, breakdowns


if __name__ == '__main__':
    main()