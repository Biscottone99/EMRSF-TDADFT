"""User-facing MRSF-TDA solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .kernels import mrsf_two_electron_components
from .reference import analyse_reference
from .ri import ResponseJK
from .solver import solve_lowest
from .space import MRSFSpace

HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class MRSFResult:
    """Vertical MRSF state-interaction eigenpairs."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    target_multiplicity: int
    space: MRSFSpace
    matrix: np.ndarray | None = None
    converged: np.ndarray | None = None
    residual_norms: np.ndarray | None = None
    iterations: int | None = None
    matvecs: int | None = None
    solver: str = "dense"
    initial_guesses: int | None = None
    symmetry_seed_counts: tuple[tuple[int, int], ...] = ()

    @property
    def energies_ev(self) -> np.ndarray:
        return self.eigenvalues * HARTREE_TO_EV

    @property
    def excitation_energies(self) -> np.ndarray:
        """Energies relative to the lowest response state (physical S0/T0)."""

        return self.eigenvalues - self.eigenvalues[0]

    @property
    def excitation_energies_ev(self) -> np.ndarray:
        return self.excitation_energies * HARTREE_TO_EV

    def state_symmetries(self):
        """Return point-group labels and squared-vector purities for all roots."""

        from .symmetry import classify_vector, mrsf_basis_irrep_ids

        basis_irreps, groupname = mrsf_basis_irrep_ids(self.space)
        return tuple(
            classify_vector(self.eigenvectors[:, state], basis_irreps, groupname)
            for state in range(self.eigenvectors.shape[1])
        )

    def state_density(self, state: int):
        """Return reference, difference, and state 1-RDMs for one root."""

        from .observables import mrsf_state_density

        return mrsf_state_density(self, state)

    def charge_transfer_descriptors(
        self,
        *,
        states=None,
        grid_level: int = 5,
        prune: bool = True,
        block_size: int = 20000,
    ):
        """Evaluate the density-difference observables of SI eqs. S42--S47."""

        from .observables import analyse_mrsf_charge_transfer

        return analyse_mrsf_charge_transfer(
            self,
            states=states,
            grid_level=grid_level,
            prune=prune,
            block_size=block_size,
        )

    def density_grid_fields(
        self,
        state: int,
        *,
        grid_level: int = 5,
        prune: bool = True,
        block_size: int = 20000,
    ):
        """Return the S42/S43 fields on the atom-centred quadrature grid."""

        from .observables import density_grid_fields

        return density_grid_fields(
            self,
            state,
            grid_level=grid_level,
            prune=prune,
            block_size=block_size,
        )

    def write_log(
        self,
        path,
        *,
        descriptors=None,
        transition_properties=None,
        title: str | None = None,
        top_configurations: int = 8,
    ):
        """Write an ORCA-like, plain-text validation report."""

        from .log import write_mrsf_log

        return write_mrsf_log(
            self,
            path,
            descriptors=descriptors,
            transition_properties=transition_properties,
            title=title,
            top_configurations=top_configurations,
        )

    def transition_properties(self, *, initial_state: int = 0, final_states=None):
        """Return rigorous length-gauge properties of MRSF transitions."""

        from .properties import analyse_mrsf_transition_properties

        return analyse_mrsf_transition_properties(
            self,
            initial_state=initial_state,
            final_states=final_states,
        )

    def save_postprocessing(
        self,
        path,
        *,
        descriptors=None,
        transition_properties=None,
        calculation_label: str | None = None,
    ):
        """Save complete states, orbital data and one-electron AO integrals."""

        from .postprocess import save_postprocessing_archive

        return save_postprocessing_archive(
            self,
            path,
            descriptors=descriptors,
            transition_properties=transition_properties,
            calculation_label=calculation_label,
        )

    def write_reference_molden(self, path):
        """Write the converged triplet-reference orbitals in Molden format."""

        from .orbitals import write_reference_molden

        return write_reference_molden(self.space.ref.mf, path)

    def natural_transition_orbitals(
        self,
        final_state: int,
        *,
        initial_state: int = 0,
    ):
        """Return MRSF natural transition orbitals for one transition."""

        from .orbitals import mrsf_natural_transition_orbitals

        return mrsf_natural_transition_orbitals(
            self,
            final_state,
            initial_state=initial_state,
        )

    def write_nto_molden(
        self,
        output_dir,
        *,
        initial_state: int = 0,
        final_states=None,
        weight_threshold: float = 1.0e-4,
        max_pairs: int | None = None,
    ):
        """Write one Molden file per selected MRSF state transition."""

        from .orbitals import write_mrsf_nto_folder

        return write_mrsf_nto_folder(
            self,
            output_dir,
            initial_state=initial_state,
            final_states=final_states,
            weight_threshold=weight_threshold,
            max_pairs=max_pairs,
        )


class MRSFTDA:
    """Collinear MRSF-TDHF/TDDFT in the Tamm-Dancoff approximation.

    Parameters
    ----------
    mf
        Converged PySCF ROHF or ROKS triplet reference.
    target_multiplicity
        ``1`` for spin-adapted singlets or ``3`` for triplets.

    Notes
    -----
    The default kernel is matrix-free Davidson.  ``solver="dense"`` remains
    available for equation-level validation on small systems.
    """

    def __init__(
        self,
        mf,
        *,
        target_multiplicity: int = 1,
        nstates: int = 6,
        spc_coco: float | None = None,
        spc_ovov: float | None = None,
        spc_coov: float | None = None,
        density_fit: bool = True,
        auxbasis=None,
        solver: str = "davidson",
        conv_tol: float = 1.0e-9,
        max_cycle: int = 80,
        max_space: int | None = None,
        progress: bool = False,
    ) -> None:
        self.ref = analyse_reference(mf)
        self.space = MRSFSpace(self.ref, target_multiplicity)
        self.target_multiplicity = target_multiplicity
        self.nstates = int(nstates)
        self.spc_coco = spc_coco
        self.spc_ovov = spc_ovov
        self.spc_coov = spc_coov
        self.jk = ResponseJK(mf, density_fit=density_fit, auxbasis=auxbasis)
        self.solver = solver
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.max_space = max_space
        self.progress = bool(progress)

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = self.space.mask_redundant(vector)
        densities = self.space.ao_components(vector)
        fock_like = mrsf_two_electron_components(
            self.ref.mf,
            densities,
            target_multiplicity=self.target_multiplicity,
            spc_coco=self.spc_coco,
            spc_ovov=self.spc_ovov,
            spc_coov=self.spc_coov,
            jk=self.jk,
        )
        return self.space.mo_action(fock_like) + self.space.one_electron_action(vector)

    def build_matrix(self, *, symmetrise: bool = True) -> np.ndarray:
        eye = np.eye(self.space.size)
        matrix = np.column_stack([self.matvec(eye[:, i]) for i in range(self.space.size)])
        if symmetrise:
            matrix = 0.5 * (matrix + matrix.T)
        return matrix

    def reduced_matvec(self, vector: np.ndarray) -> np.ndarray:
        live = np.isfinite(self.space.diagonal_guess())
        full = np.zeros(self.space.size)
        full[live] = vector
        return self.matvec(full)[live]

    def kernel(self, *, solver: str | None = None) -> MRSFResult:
        solver = self.solver if solver is None else solver
        if solver.lower() != "dense":
            from .symmetry import mrsf_basis_irrep_ids

            live = np.isfinite(self.space.diagonal_guess())
            diagonal = self.space.diagonal_guess()[live]
            basis_irreps, _ = mrsf_basis_irrep_ids(self.space)
            iterative = solve_lowest(
                self.reduced_matvec,
                diagonal,
                solver=solver,
                nroots=min(self.nstates, len(diagonal)),
                tol=self.conv_tol,
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

        matrix = self.build_matrix()
        diagonal = self.space.diagonal_guess()
        live = np.isfinite(diagonal)
        reduced = matrix[np.ix_(live, live)]
        values, vectors_reduced = eigh(reduced, subset_by_index=(0, min(self.nstates, len(reduced)) - 1))
        vectors = np.zeros((self.space.size, len(values)))
        vectors[live] = vectors_reduced
        return MRSFResult(
            eigenvalues=values,
            eigenvectors=vectors,
            target_multiplicity=self.target_multiplicity,
            space=self.space,
            matrix=matrix,
            converged=np.ones(len(values), dtype=bool),
            residual_norms=np.zeros(len(values)),
            solver="dense",
        )
