"""Numerically tightened drivers built on the unchanged MRSF operator."""

from __future__ import annotations

import numpy as np

from .mrsf import MRSFResult, MRSFTDA
from .solver import solve_lowest
from .symmetry import mrsf_basis_irrep_ids


class HighAccuracyMRSFTDA(MRSFTDA):
    """MRSF-TDA with an explicit Davidson residual threshold.

    The MRSF response space, AO kernels, spin adaptation, SPC coefficients and
    matrix-vector product are inherited verbatim from :class:`MRSFTDA`.  This
    class changes only the numerical stopping criterion of the iterative
    diagonalisation, so MRSF and EMRSF can be compared with the same strict
    residual target in the high-accuracy driver.
    """

    def __init__(self, *args, residual_tol: float = 1.0e-9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_tol = float(residual_tol)
        if self.residual_tol <= 0.0:
            raise ValueError("residual_tol must be positive")

    def kernel(self, *, solver: str | None = None) -> MRSFResult:
        selected_solver = self.solver if solver is None else solver
        if selected_solver.lower() == "dense":
            return super().kernel(solver="dense")

        live = np.isfinite(self.space.diagonal_guess())
        diagonal = self.space.diagonal_guess()[live]
        basis_irreps, _ = mrsf_basis_irrep_ids(self.space)
        iterative = solve_lowest(
            self.reduced_matvec,
            diagonal,
            solver=selected_solver,
            nroots=min(self.nstates, len(diagonal)),
            tol=self.conv_tol,
            residual_tol=self.residual_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            max_memory=getattr(self.ref.mf, "max_memory", 4000),
            verbose=max(0, getattr(self.ref.mf, "verbose", 0) - 2),
            progress=self.progress,
            progress_label="MRSF Davidson",
            basis_irreps=basis_irreps[live],
        )
        vectors = np.zeros((self.space.size, len(iterative.eigenvalues)))
        vectors[live] = iterative.eigenvectors
        return MRSFResult(
            eigenvalues=iterative.eigenvalues,
            eigenvectors=vectors,
            target_multiplicity=self.target_multiplicity,
            space=self.space,
            matrix=None,
            converged=iterative.converged,
            residual_norms=iterative.residual_norms,
            iterations=iterative.iterations,
            matvecs=iterative.matvecs,
            solver=iterative.solver,
            initial_guesses=iterative.initial_guesses,
            symmetry_seed_counts=iterative.symmetry_seed_counts,
        )
