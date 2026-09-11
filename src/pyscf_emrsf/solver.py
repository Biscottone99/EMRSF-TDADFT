"""Iterative symmetric eigensolvers for matrix-free response operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from time import perf_counter
import warnings

import numpy as np
from pyscf import lib
from scipy.sparse.linalg import LinearOperator, lobpcg


@dataclass(frozen=True)
class IterativeEigensolution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    converged: np.ndarray
    residual_norms: np.ndarray
    iterations: int
    matvecs: int
    solver: str
    initial_guesses: int
    symmetry_seed_counts: tuple[tuple[int, int], ...]


def _guess_indices(
    diagonal: np.ndarray,
    nroots: int,
    basis_irreps: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Choose diagonal seeds without losing an invariant symmetry block.

    A Davidson Krylov space cannot discover a point-group block that is absent
    from its initial vectors.  More subtly, a diagonal (or nearly diagonal)
    block seeded with fewer than ``nroots`` independent unit vectors can hide a
    low state even when another state of the same irrep is present.  The
    published benchmarks use Abelian symmetry, so retain the ordinary global
    oversampling and also seed up to ``nroots`` lowest diagonals in *each*
    irrep.  In C1 this reduces to the ordinary global choice.
    """

    diagonal = np.asarray(diagonal, dtype=float)
    order = np.argsort(diagonal, kind="stable")
    nguess = min(len(order), max(2 * nroots, nroots + 4, 2))
    selected = set(map(int, order[:nguess]))

    if basis_irreps is None:
        counts = ((0, len(selected)),)
    else:
        irreps = np.asarray(basis_irreps, dtype=int)
        if irreps.shape != diagonal.shape:
            raise ValueError("basis_irreps must have the same shape as diagonal")
        for irrep in np.unique(irreps):
            block = np.flatnonzero(irreps == irrep)
            block_order = block[np.argsort(diagonal[block], kind="stable")]
            selected.update(map(int, block_order[: min(nroots, len(block_order))]))
        counts = tuple(
            (int(irrep), sum(int(irreps[index]) == int(irrep) for index in selected))
            for irrep in np.unique(irreps)
        )

    indices = np.asarray(sorted(selected, key=lambda index: (diagonal[index], index)))
    return indices, counts


def _unit_guesses(
    diagonal: np.ndarray,
    nroots: int,
    basis_irreps: np.ndarray | None = None,
) -> tuple[list[np.ndarray], tuple[tuple[int, int], ...]]:
    indices, counts = _guess_indices(diagonal, nroots, basis_irreps)
    guesses: list[np.ndarray] = []
    for index in indices:
        vector = np.zeros(len(diagonal))
        vector[index] = 1.0
        guesses.append(vector)
    return guesses, counts


def davidson_lowest(
    matvec: Callable[[np.ndarray], np.ndarray],
    diagonal: np.ndarray,
    *,
    nroots: int,
    tol: float = 1.0e-9,
    residual_tol: float | None = None,
    max_cycle: int = 80,
    max_space: int | None = None,
    max_memory: float = 4000,
    verbose: int = 0,
    progress: bool = False,
    progress_label: str = "Davidson",
    basis_irreps: np.ndarray | None = None,
) -> IterativeEigensolution:
    """Compute the lowest roots with PySCF's restarted Davidson solver."""

    diagonal = np.asarray(diagonal, dtype=float)
    dimension = len(diagonal)
    if dimension == 0:
        raise ValueError("Cannot diagonalise an empty response space")
    nroots = min(int(nroots), dimension)
    guesses, seed_counts = _unit_guesses(diagonal, nroots, basis_irreps)
    minimum_space = min(dimension, len(guesses) + nroots)
    if max_space is None:
        # MRSF/EMRSF preconditioners are only approximate; a somewhat broader
        # restarted subspace avoids losing a low root whose orbital-energy
        # diagonal is interleaved with another configuration class.
        max_space = min(
            dimension,
            max(40, 8 * nroots + 12, minimum_space),
        )
    else:
        max_space = min(dimension, max(int(max_space), minimum_space))
    if residual_tol is None:
        # PySCF otherwise defaults to sqrt(tol), which can accept a root from
        # an incomplete response subspace before a lower, strongly mixed root
        # enters the Ritz space.
        residual_tol = max(1.0e-7, 10.0 * tol)

    counters = {"matvecs": 0, "iterations": 0}
    start_time = perf_counter()

    def aop(vectors):
        output = []
        for vector in vectors:
            output.append(np.asarray(matvec(np.asarray(vector))))
            counters["matvecs"] += 1
        return output

    def precondition(residual, eigenvalue, _vector):
        denominator = diagonal - eigenvalue
        small = np.abs(denominator) < 1.0e-8
        denominator = denominator.copy()
        denominator[small] = np.where(denominator[small] < 0.0, -1.0e-8, 1.0e-8)
        return residual / denominator

    def callback(environment):
        counters["iterations"] = int(environment.get("icyc", 0)) + 1
        if progress:
            values = np.asarray(environment.get("e", []), dtype=float)
            residuals = np.asarray(environment.get("dx_norm", []), dtype=float)
            subspace = int(environment.get("space", 0))
            max_residual = float(np.max(residuals)) if residuals.size else np.nan
            elapsed = perf_counter() - start_time
            roots = " ".join(f"{value: .8f}" for value in values)
            print(
                f"[{progress_label}] iter={counters['iterations']:3d} "
                f"subspace={subspace:4d} matvecs={counters['matvecs']:4d} "
                f"max|r|={max_residual:.3e} elapsed={elapsed / 60.0:.1f} min "
                f"roots/Eh=[{roots}]",
                flush=True,
            )

    converged, values, vectors = lib.davidson1(
        aop,
        guesses,
        precondition,
        tol=tol,
        tol_residual=residual_tol,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory=max_memory,
        nroots=nroots,
        callback=callback,
        verbose=verbose,
    )
    coefficient = np.column_stack(vectors)
    residuals = np.asarray(
        [np.linalg.norm(matvec(coefficient[:, k]) - values[k] * coefficient[:, k])
         for k in range(len(values))]
    )
    counters["matvecs"] += len(values)
    converged = np.asarray(converged, dtype=bool) & (
        residuals <= 1.1 * residual_tol
    )
    if counters["iterations"] == 0:
        counters["iterations"] = 1
    return IterativeEigensolution(
        eigenvalues=np.asarray(values),
        eigenvectors=coefficient,
        converged=converged,
        residual_norms=residuals,
        iterations=counters["iterations"],
        matvecs=counters["matvecs"],
        solver="davidson",
        initial_guesses=len(guesses),
        symmetry_seed_counts=seed_counts,
    )


def lobpcg_lowest(
    matvec: Callable[[np.ndarray], np.ndarray],
    diagonal: np.ndarray,
    *,
    nroots: int,
    tol: float = 1.0e-9,
    max_cycle: int = 80,
    basis_irreps: np.ndarray | None = None,
) -> IterativeEigensolution:
    """LOBPCG alternative, primarily for solver benchmarking."""

    diagonal = np.asarray(diagonal, dtype=float)
    dimension = len(diagonal)
    if dimension == 0:
        raise ValueError("Cannot diagonalise an empty response space")
    nroots = min(int(nroots), dimension)
    counters = {"matvecs": 0}

    def apply(array):
        array = np.asarray(array)
        if array.ndim == 1:
            counters["matvecs"] += 1
            return matvec(array)
        counters["matvecs"] += array.shape[1]
        return np.column_stack([matvec(array[:, k]) for k in range(array.shape[1])])

    operator = LinearOperator((dimension, dimension), matvec=apply, matmat=apply)
    # As for Davidson, oversample the orbital-energy guesses.  LOBPCG preserves
    # its block size, so solve the larger block and retain only the requested
    # lowest roots afterwards.
    guess_vectors, seed_counts = _unit_guesses(diagonal, nroots, basis_irreps)
    guesses = np.column_stack(guess_vectors)
    # Oversampled auxiliary roots may remain above tolerance even when every
    # requested root is converged.  Report convergence from the requested
    # residuals below instead of emitting SciPy's block-wide warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values, vectors, residual_history = lobpcg(
            operator,
            guesses,
            largest=False,
            tol=tol,
            maxiter=max_cycle,
            retResidualNormsHistory=True,
        )
    order = np.argsort(values)[:nroots]
    values = np.asarray(values)[order]
    vectors = np.asarray(vectors)[:, order]
    residuals = np.asarray(
        [np.linalg.norm(matvec(vectors[:, k]) - values[k] * vectors[:, k])
         for k in range(len(values))]
    )
    counters["matvecs"] += len(values)
    return IterativeEigensolution(
        eigenvalues=values,
        eigenvectors=vectors,
        converged=residuals <= max(1.0e-7, 10.0 * tol),
        residual_norms=residuals,
        iterations=max(1, len(residual_history) - 1),
        matvecs=counters["matvecs"],
        solver="lobpcg",
        initial_guesses=guesses.shape[1],
        symmetry_seed_counts=seed_counts,
    )


def solve_lowest(matvec, diagonal, *, solver: str = "davidson", **kwargs):
    name = solver.lower().replace("-", "")
    if name == "davidson":
        return davidson_lowest(matvec, diagonal, **kwargs)
    if name == "lobpcg":
        allowed = {
            key: kwargs[key]
            for key in ("nroots", "tol", "max_cycle", "basis_irreps")
            if key in kwargs
        }
        return lobpcg_lowest(matvec, diagonal, **allowed)
    raise ValueError("solver must be 'davidson' or 'lobpcg'")
