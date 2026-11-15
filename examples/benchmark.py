"""Benchmark of the sparse solvers against dense and SciPy direct solves.

An end-loaded cantilever is solved on four Quad8 grids, each with four
times the degrees of freedom of the one before. On every grid the reduced
system K_ff u = F_rhs is solved by

- a dense direct solve (NumPy), skipped once K_ff is too large to store densely,
- a sparse direct solve (SciPy spsolve), used as the reference answer,
- conjugate gradients with no preconditioner, Jacobi and IC(0), each run
  twice: with the pure-Python kernels and with the C++ kernels
  (backend="cpp"), which replace the matrix-vector product and the IC(0)
  triangular solves.

Solve time, CG iterations, storage of K_ff and the error of each answer
against the reference are recorded. A second pass times each stage of the
IC(0) path on its own, including one matvec and one forward plus backward
solve in each backend, and checks for both backends that the stages add
up to the measured IC(0) solve time.

Three tables are printed (Python backend, C++ backend with its speed-up
and its gap to SciPy, stage breakdown) and three log-log plots are saved
to docs/images/.
The largest grid takes a few minutes; run from any directory with

    python examples/benchmark.py
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve

from fesolver import _kernels
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
CG_RUNS = [('No Preconditioner', None), ('Jacobi', 'jacobi'), ('IC(0)', 'ic0')]
BACKENDS = [('', 'python'), (' C++', 'cpp')]   # key suffix, backend name


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
        dense and the six CG runs their error against SciPy; CG iteration
        counts; storage of K_ff in MB, dense and CSR. Keys of the C++ runs
        end in ' C++' before the quantity, e.g. 'IC(0) C++ Time'. Dense time
        and error are None above DENSE_LIMIT free DOFs, where the dense solve
        is skipped. Dense storage is always computed.
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

    for name, preconditioner in CG_RUNS:
        for suffix, backend in BACKENDS:
            key = f'{name}{suffix}'
            (u, history), result[f'{key} Time'] = timed(
                lambda: cg(K_ff, F_rhs, preconditioner=preconditioner, backend=backend))
            result[f'{key} Iterations'] = len(history) - 1
            result[f'{key} Error'] = relative_error(u, u_reference)

    return result


def stage_breakdown(nx, ny):
    """Time each stage of the IC(0) path on one grid.

    Stages that run once per solve (element loop, CSR build, partitioning,
    IC(0) factorisation, transpose) are timed once. Stages that run once
    per CG iteration (one matvec, one forward plus backward solve) are
    averaged over REPEATS calls, because a single call is too short to time
    reliably, and are timed for both the Python and the C++ kernels on the
    same arrays. The input vector is all ones: the cost depends on the
    sparsity pattern, not on the values.

    Parameters
    ----------
    nx, ny : int
        Number of elements along x and y.

    Returns
    -------
    dict
        Grid name and the time of each stage in seconds. 'Matvec' and
        'Solve' are the Python kernels, 'C++ Matvec' and 'C++ Solve' the
        C++ ones.
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

    start = time.perf_counter()
    for _ in range(REPEATS):
        _kernels.matvec(K_ff.indptr, K_ff.indices, K_ff.data, x)
    stages['C++ Matvec'] = (time.perf_counter() - start) / REPEATS

    start = time.perf_counter()
    for _ in range(REPEATS):
        y = _kernels.forward_solve(L.indptr, L.indices, L.data, x)
        _kernels.backward_solve(L_transposed.indptr, L_transposed.indices,
                                L_transposed.data, y)
    stages['C++ Solve'] = (time.perf_counter() - start) / REPEATS

    return stages


def check_breakdown(stages, result, rtol=0.2):
    """Check that the timed stages add up to the measured IC(0) solve times.

    The predicted time is the one-off setup (factorisation and transpose)
    plus the number of iterations times the cost of one iteration (one
    matvec and one forward plus backward solve). It is formed twice, once
    with the Python per-call times and the Python iteration count, and once
    with the C++ ones. The vector updates and dot products inside CG are
    not timed, so the prediction runs slightly low on small grids.

    Parameters
    ----------
    stages : dict
        Output of stage_breakdown for one grid. Updated in place with
        'Predicted', 'Ratio' (predicted / measured) and 'Check' ('ok' if
        within rtol of the measured time, otherwise 're-run'), and the
        same three keys prefixed with 'C++ ' for the C++ path.
    result : dict
        Output of compare_solvers for the same grid.
    rtol : float, optional
        Relative tolerance. Defaults to 0.2.
    """
    setup = stages['IC(0) Factorisation'] + stages['Transpose']
    for prefix, suffix in [('', ''), ('C++ ', ' C++')]:
        measured = result[f'IC(0){suffix} Time']
        per_iteration = stages[f'{prefix}Matvec'] + stages[f'{prefix}Solve']
        predicted = setup + result[f'IC(0){suffix} Iterations'] * per_iteration
        stages[f'{prefix}Predicted'] = predicted
        stages[f'{prefix}Ratio'] = predicted / measured
        stages[f'{prefix}Check'] = 'ok' if np.isclose(measured, predicted, rtol=rtol) else 're-run'


def print_tables(results, breakdowns):
    """Print the solver comparison and the stage breakdown, one row per grid.

    Three tables: the Python backend with the direct solvers and storage;
    the C++ backend with the speed-up of its IC(0) solve over the Python
    one and its gap to SciPy (C++ IC(0) time / SciPy time); and the stage
    breakdown with per-call kernel speed-ups and both stage checks.
    """
    def fmt(value):
        return '-' if value is None else f'{value:.3g}'

    def cg_cols(r, suffix):
        return [f"{fmt(r[f'{name}{suffix} Time'])} ({r[f'{name}{suffix} Iterations']})"
                for name, _ in CG_RUNS]

    print('\nPython backend: solve time (s), CG iterations in brackets, storage of K_ff (MB)')
    print(f"{'Grid':>7} {'Free DOFs':>9} {'Dense':>8} {'SciPy':>8} "
          f"{'CG':>13} {'Jacobi':>13} {'IC(0)':>13} {'Dense MB':>9} {'CSR MB':>7}")
    for r in results:
        cols = cg_cols(r, '')
        print(f"{r['Grid']:>7} {r['Free DOFs']:>9} {fmt(r['Dense Time']):>8} "
              f"{fmt(r['SciPy Time']):>8} {cols[0]:>13} {cols[1]:>13} "
              f"{cols[2]:>13} {fmt(r['Dense Memory']):>9} {fmt(r['CSR Memory']):>7}")

    print('\nC++ backend: solve time (s), CG iterations in brackets')
    print(f"{'Grid':>7} {'Free DOFs':>9} {'CG':>13} {'Jacobi':>13} {'IC(0)':>13} "
          f"{'IC(0) speed-up':>14} {'Gap to SciPy':>12}")
    for r in results:
        cols = cg_cols(r, ' C++')
        speed_up = r['IC(0) Time'] / r['IC(0) C++ Time']
        gap = r['IC(0) C++ Time'] / r['SciPy Time']
        print(f"{r['Grid']:>7} {r['Free DOFs']:>9} {cols[0]:>13} {cols[1]:>13} "
              f"{cols[2]:>13} {speed_up:>13.1f}x {gap:>11.0f}x")

    error_keys = ['Dense'] + [f'{name}{suffix}' for name, _ in CG_RUNS
                              for suffix, _ in BACKENDS]
    worst = max(r[f'{k} Error'] for r in results for k in error_keys
                if r[f'{k} Error'] is not None)
    print(f'Largest relative error against SciPy spsolve: {worst:.1e}')

    keys = ['Element Loop', 'CSR Build', 'Partitioning', 'IC(0) Factorisation',
            'Transpose', 'Matvec', 'C++ Matvec', 'Solve', 'C++ Solve']
    print('\nStage breakdown (s); Matvec and Solve are per call')
    print(f"{'Grid':>7} " + ' '.join(f'{k:>{max(len(k), 8)}}' for k in keys))
    for s in breakdowns:
        print(f"{s['Grid']:>7} " + ' '.join(f'{fmt(s[k]):>{max(len(k), 8)}}' for k in keys))

    print('\nKernel speed-up per call (Python / C++) and stage check (predicted / measured IC(0) time)')
    print(f"{'Grid':>7} {'Matvec':>8} {'Solve':>8} {'Python ratio':>12} {'Check':>6} "
          f"{'C++ ratio':>9} {'Check':>6}")
    for s in breakdowns:
        print(f"{s['Grid']:>7} {s['Matvec'] / s['C++ Matvec']:>7.0f}x "
              f"{s['Solve'] / s['C++ Solve']:>7.0f}x {s['Ratio']:>12.2f} {s['Check']:>6} "
              f"{s['C++ Ratio']:>9.2f} {s['C++ Check']:>6}")


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
                   ('CG, IC(0), C++ kernels', column('IC(0) C++ Time')),
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