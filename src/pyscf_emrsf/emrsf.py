"""Extended MRSF-TDDFT/TDA singlet and triplet enlarged spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .coupling import DirectCouplingOperator
from .kernels import mrsf_two_electron_components
from .lr import CVSpace, CorrectedLRTDA
from .mrsf import HARTREE_TO_EV, MRSFResult, MRSFTDA
from .ri import RICouplingOperator
from .resources import estimate_emrsf_resources
from .solver import solve_lowest
from .space import MRSFSpace


@dataclass(frozen=True)
class EMRSFStateMatch:
    """Term-constrained MRSF correlation for an EMRSF root.

    ``mrsf_state`` is ``None`` when the same spectroscopic term (for example,
    ``2 B1``) is not present among the computed MRSF roots.
    ``candidate_mrsf_state`` retains the best-overlap same-symmetry root as a
    diagnostic; it is never substituted for a missing term correlation.
    """

    emrsf_state: int
    mrsf_state: int | None
    candidate_mrsf_state: int | None
    overlap_squared: float
    conditional_overlap_squared: float
    raw_energy_shift_hartree: float
    excitation_energy_shift_hartree: float
    status: str

    @property
    def matched(self) -> bool:
        return self.mrsf_state is not None

    @property
    def raw_energy_shift_ev(self) -> float:
        return float(self.raw_energy_shift_hartree * HARTREE_TO_EV)

    @property
    def excitation_energy_shift_ev(self) -> float:
        return float(self.excitation_energy_shift_hartree * HARTREE_TO_EV)


@dataclass(frozen=True)
class EMRSFStateDecomposition:
    """Expectation-value closure of one eigenstate in the enlarged space."""

    state: int
    eigenvalue_hartree: float
    mrsf_weight: float
    cv_weight: float
    mrsf_block_hartree: float
    cv_block_hartree: float
    coupling_hartree: float
    reconstructed_hartree: float
    closure_error_hartree: float

    @property
    def closure_error_ev(self) -> float:
        return float(self.closure_error_hartree * HARTREE_TO_EV)


@dataclass(frozen=True)
class EMRSFResult:
    """Eigenpairs and CV-sector diagnostics for an EMRSF calculation."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    matrix: np.ndarray | None
    mrsf_size: int
    cv_space: CVSpace
    mrsf_result: MRSFResult | None
    live_mrsf_indices: np.ndarray
    converged: np.ndarray | None = None
    residual_norms: np.ndarray | None = None
    iterations: int | None = None
    matvecs: int | None = None
    solver: str = "dense"
    initial_guesses: int | None = None
    symmetry_seed_counts: tuple[tuple[int, int], ...] = ()
    exact_exchange: float | None = None
    coupling_scale: float | None = None
    ground_offset_hartree: float | None = None
    ground_offset_operator_hartree: float | None = None
    ground_offset_source_error_hartree: float | None = None
    coupling_frobenius_norm: float | None = None
    mrsf_space_object: MRSFSpace | None = None
    direct_emrsf: bool = False
    integral_backend: str = "density-fitting"
    state_decompositions: tuple[EMRSFStateDecomposition, ...] = ()

    @property
    def mrsf_space(self) -> MRSFSpace:
        if self.mrsf_space_object is not None:
            return self.mrsf_space_object
        if self.mrsf_result is None:
            raise RuntimeError("The MRSF basis was not retained in this result")
        return self.mrsf_result.space

    @property
    def energies_ev(self) -> np.ndarray:
        return self.eigenvalues * HARTREE_TO_EV

    @property
    def target_multiplicity(self) -> int:
        """Spin multiplicity represented by the enlarged response space."""

        return int(self.mrsf_space.target_multiplicity)

    @property
    def excitation_energies(self) -> np.ndarray:
        """Energies relative to the lowest state in the same spin manifold."""

        return self.eigenvalues - self.eigenvalues[0]

    @property
    def excitation_energies_ev(self) -> np.ndarray:
        return self.excitation_energies * HARTREE_TO_EV

    def energies_from_origin(self, origin_hartree: float) -> np.ndarray:
        """Return all roots relative to an explicitly supplied common origin."""

        return self.eigenvalues - float(origin_hartree)

    def energies_from_origin_ev(self, origin_hartree: float) -> np.ndarray:
        """Return common-origin energies in electronvolts."""

        return self.energies_from_origin(origin_hartree) * HARTREE_TO_EV

    @property
    def cv_weights(self) -> np.ndarray:
        """The paper's :math:`\\gamma^K_{CV}` for every EMRSF state."""

        return np.sum(self.eigenvectors[self.mrsf_size :, :] ** 2, axis=0)

    @property
    def mrsf_weights(self) -> np.ndarray:
        return 1.0 - self.cv_weights

    def state_symmetries(self):
        """Return point-group labels and squared-vector purities for all roots."""

        from .symmetry import (
            classify_vector,
            cv_basis_irrep_ids,
            mrsf_basis_irrep_ids,
        )

        mrsf_irreps, mrsf_group = mrsf_basis_irrep_ids(self.mrsf_space)
        cv_irreps, cv_group = cv_basis_irrep_ids(self.cv_space)
        if mrsf_group != cv_group:
            raise ValueError("MRSF and LR sectors use different point groups")
        basis_irreps = np.concatenate(
            (mrsf_irreps[self.live_mrsf_indices], cv_irreps)
        )
        return tuple(
            classify_vector(self.eigenvectors[:, state], basis_irreps, mrsf_group)
            for state in range(self.eigenvectors.shape[1])
        )

    def state_matches(
        self,
        *,
        low_overlap_warning: float = 0.5,
    ) -> tuple[EMRSFStateMatch, ...]:
        """Correlate equal spectroscopic terms between MRSF and EMRSF.

        This diagnostic pairs equal ordinal-within-irrep labels (``2 B1`` with
        ``2 B1``).  The article identifies benchmark states by term but does not
        define an automatic numerical root-tracking algorithm.  Raw and
        conditional overlaps are therefore also reported; a low overlap is
        flagged and never silently redirects the term match.
        """

        threshold = float(low_overlap_warning)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("low_overlap_warning must lie between 0 and 1")
        if self.mrsf_result is None:
            raise RuntimeError(
                "MRSF state matching requires solve_mrsf_first=True; the direct "
                "EMRSF calculation intentionally did not diagonalise MRSF"
            )

        mrsf_vectors = self.mrsf_result.eigenvectors[self.live_mrsf_indices, :]
        emrsf_mrsf = self.eigenvectors[: self.mrsf_size, :]
        overlaps = np.abs(mrsf_vectors.T @ emrsf_mrsf) ** 2
        sector_weights = np.asarray(self.mrsf_weights, dtype=float)
        conditional = np.zeros_like(overlaps)
        nonzero = sector_weights > 1.0e-14
        conditional[:, nonzero] = overlaps[:, nonzero] / sector_weights[nonzero]

        mrsf_symmetries = self.mrsf_result.state_symmetries()
        emrsf_symmetries = self.state_symmetries()
        same_symmetry = np.asarray(
            [
                [
                    mr.irrep_id == em.irrep_id
                    for em in emrsf_symmetries
                ]
                for mr in mrsf_symmetries
            ],
            dtype=bool,
        )

        from .symmetry import term_labels

        mrsf_terms = term_labels(mrsf_symmetries)
        emrsf_terms = term_labels(emrsf_symmetries)
        mrsf_by_term = {term: state for state, term in enumerate(mrsf_terms)}

        matches = []
        for emrsf_state, emrsf_term in enumerate(emrsf_terms):
            symmetry_candidates = np.flatnonzero(same_symmetry[:, emrsf_state])
            if symmetry_candidates.size:
                best_local = int(
                    np.argmax(conditional[symmetry_candidates, emrsf_state])
                )
                candidate = int(symmetry_candidates[best_local])
                candidate_overlap = float(overlaps[candidate, emrsf_state])
                candidate_conditional = float(conditional[candidate, emrsf_state])
            else:
                candidate = None
                candidate_overlap = 0.0
                candidate_conditional = 0.0

            accepted_state = mrsf_by_term.get(emrsf_term)
            if accepted_state is not None:
                accepted_state = int(accepted_state)
                overlap = float(overlaps[accepted_state, emrsf_state])
                conditional_overlap = float(
                    conditional[accepted_state, emrsf_state]
                )
                raw_shift = float(
                    self.eigenvalues[emrsf_state]
                    - self.mrsf_result.eigenvalues[accepted_state]
                )
                excitation_shift = float(
                    self.excitation_energies[emrsf_state]
                    - self.mrsf_result.excitation_energies[accepted_state]
                )
                status = (
                    "MATCHED_TERM"
                    if conditional_overlap >= threshold
                    else "MATCHED_TERM_LOW_OVERLAP"
                )
                mrsf_state = accepted_state
            else:
                overlap = candidate_overlap
                conditional_overlap = candidate_conditional
                raw_shift = np.nan
                excitation_shift = np.nan
                mrsf_state = None
                status = "UNMATCHED_TERM_NOT_COMPUTED"

            matches.append(
                EMRSFStateMatch(
                    emrsf_state=emrsf_state,
                    mrsf_state=mrsf_state,
                    candidate_mrsf_state=candidate,
                    overlap_squared=overlap,
                    conditional_overlap_squared=conditional_overlap,
                    raw_energy_shift_hartree=raw_shift,
                    excitation_energy_shift_hartree=excitation_shift,
                    status=status,
                )
            )
        return tuple(matches)

    def charge_transfer_descriptors(
        self,
        *,
        states=None,
        grid_level: int = 5,
        prune: bool = True,
        block_size: int = 20000,
    ):
        """Evaluate S42--S47 for the underlying matched MRSF states."""

        if self.mrsf_result is None:
            raise RuntimeError(
                "S42--S47 in this release are MRSF-state observables and are "
                "unavailable when MRSF pre-diagonalisation is disabled"
            )

        return self.mrsf_result.charge_transfer_descriptors(
            states=states,
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
        coupling_audit=None,
        lr_cv_audit=None,
        title: str | None = None,
        top_configurations: int = 8,
    ):
        """Write the full ORCA-like EMRSF validation report."""

        from .log import write_emrsf_log

        return write_emrsf_log(
            self,
            path,
            descriptors=descriptors,
            transition_properties=transition_properties,
            coupling_audit=coupling_audit,
            lr_cv_audit=lr_cv_audit,
            title=title,
            top_configurations=top_configurations,
        )

    def save_postprocessing(
        self,
        path,
        *,
        descriptors=None,
        transition_properties=None,
        calculation_label: str | None = None,
    ):
        """Save complete enlarged-space roots and all post-processing maps."""

        from .postprocess import save_postprocessing_archive

        return save_postprocessing_archive(
            self,
            path,
            descriptors=descriptors,
            transition_properties=transition_properties,
            calculation_label=calculation_label,
        )

    def write_eigenvectors_csv(
        self, path, *, coefficient_threshold: float = 0.0
    ):
        """Write every EMRSF root with MRSF/LR-CV row labels."""

        from .postprocess import write_eigenvectors_csv

        return write_eigenvectors_csv(
            self, path, coefficient_threshold=coefficient_threshold
        )

    def write_reference_molden(self, path):
        """Write the converged triplet-reference orbitals in Molden format."""

        from .orbitals import write_reference_molden

        return write_reference_molden(self.mrsf_space.ref.mf, path)

    def write_mrsf_nto_molden(
        self,
        output_dir,
        *,
        initial_state: int = 0,
        final_states=None,
        weight_threshold: float = 1.0e-4,
        max_pairs: int | None = None,
    ):
        """Export state-to-state NTOs of the underlying MRSF roots.

        These are deliberately named MRSF NTOs.  A full EMRSF transition 1-RDM
        would additionally require LR-LR and MRSF-LR cross-sector density terms
        that are not defined by the S42--S47 state-density construction.
        """

        from .orbitals import write_mrsf_nto_folder

        if self.mrsf_result is None:
            raise RuntimeError(
                "MRSF NTOs require solve_mrsf_first=True; full EMRSF NTOs are "
                "not defined by the published one-particle density equations"
            )

        return write_mrsf_nto_folder(
            self.mrsf_result,
            output_dir,
            initial_state=initial_state,
            final_states=final_states,
            weight_threshold=weight_threshold,
            max_pairs=max_pairs,
        )


class EMRSFTDA:
    """Corrected EMRSF-TDHF/TDDFT in the Tamm--Dancoff approximation.

    For singlets this class implements the published block matrix of Oh
    *et al.* using only PySCF: conventional MRSF, the additional LR C->V
    sector, the S30 Fock correction, and the Slater--Condon coupling of SI
    Tables S1--S3.  For triplets it applies the corresponding spin-adapted
    S1+S2 extension derived from the same determinant rows.

    Matrix-free Davidson is available with either density-fitted J/K or exact
    four-centre integral contractions.  ``solve_mrsf_first=False`` performs the
    enlarged EMRSF diagonalisation directly; the MRSF block is still evaluated
    exactly as in v0.4.1 but its standalone eigenproblem is not solved.
    """

    def __init__(
        self,
        mf,
        *,
        target_multiplicity: int = 1,
        nstates: int = 6,
        apply_fock_correction: bool = True,
        spc_coco: float | None = None,
        spc_ovov: float | None = None,
        spc_coov: float | None = None,
        density_fit: bool = True,
        auxbasis=None,
        solver: str = "davidson",
        conv_tol: float = 1.0e-9,
        residual_tol: float | None = None,
        max_cycle: int = 80,
        max_space: int | None = None,
        progress: bool = False,
        coupling_scale: float | None = None,
        solve_mrsf_first: bool = True,
    ) -> None:
        self.target_multiplicity = int(target_multiplicity)
        if self.target_multiplicity not in (1, 3):
            raise ValueError("target_multiplicity must be 1 or 3")
        self.nstates = int(nstates)
        self.mrsf = MRSFTDA(
            mf,
            target_multiplicity=self.target_multiplicity,
            nstates=nstates,
            spc_coco=spc_coco,
            spc_ovov=spc_ovov,
            spc_coov=spc_coov,
            density_fit=density_fit,
            auxbasis=auxbasis,
            solver=solver,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            max_space=max_space,
            progress=progress,
        )
        self.lr = CorrectedLRTDA(
            self.mrsf.ref,
            target_multiplicity=self.target_multiplicity,
            apply_correction=apply_fock_correction,
            response_jk=self.mrsf.jk,
        )
        self.coupling_scale = (
            float(self.lr.exact_exchange)
            if coupling_scale is None
            else float(coupling_scale)
        )
        if not np.isfinite(self.coupling_scale):
            raise ValueError("coupling_scale must be finite")
        if self.mrsf.jk.with_df is None:
            self.coupling = DirectCouplingOperator(
                self.mrsf.space,
                self.lr.space,
                self.lr.fock_mo,
                coupling_scale=self.coupling_scale,
            )
        else:
            self.coupling = RICouplingOperator(
                self.mrsf.space,
                self.lr.space,
                self.lr.fock_mo,
                coupling_scale=self.coupling_scale,
                jk=self.mrsf.jk,
            )
        self.integral_backend = self.coupling.backend
        self.solver = solver
        self.conv_tol = float(conv_tol)
        self.residual_tol = (
            None if residual_tol is None else float(residual_tol)
        )
        if self.residual_tol is not None and self.residual_tol <= 0.0:
            raise ValueError("residual_tol must be positive")
        self.max_cycle = int(max_cycle)
        self.max_space = max_space
        self.progress = bool(progress)
        self.solve_mrsf_first = bool(solve_mrsf_first)
        self.live = np.flatnonzero(np.isfinite(self.mrsf.space.diagonal_guess()))
        ref = self.mrsf.ref
        # A_G is the published singlet ground-state offset of SI eq. S23.
        # The G vector is absent from a live triplet MRSF space, so evaluate
        # this independent operator path in a temporary singlet packing while
        # leaving the requested MRSF operator untouched.
        ground_space = MRSFSpace(ref, 1)
        labels = {pair: k for k, pair in enumerate(ground_space.labels)}
        ground = labels[(ref.o2, ref.o1)]
        unit = np.zeros(ground_space.size)
        unit[ground] = 1.0
        densities = ground_space.ao_components(unit)
        fock_like = mrsf_two_electron_components(
            ref.mf,
            densities,
            target_multiplicity=1,
            spc_coco=self.mrsf.spc_coco,
            spc_ovov=self.mrsf.spc_ovov,
            spc_coov=self.mrsf.spc_coov,
            jk=self.mrsf.jk,
        )
        ground_action = ground_space.mo_action(fock_like)
        ground_action += ground_space.one_electron_action(unit)
        self.ground_offset_operator = float(ground_action[ground])

        # SI eq. S23.  Build A_G^MRSF directly instead of obtaining it from a
        # particular response-vector path.  The Coulomb integral is evaluated
        # through the same J/K backend as the response operator, so the two
        # independent constructions must agree to numerical precision.
        c_o1 = ref.coeff[:, ref.o1]
        d_o1 = np.outer(c_o1, c_o1)
        vj_o1, _ = self.mrsf.jk.get_jk(d_o1, hermi=1)
        c_o2 = ref.coeff[:, ref.o2]
        eri_o2o2_o1o1 = float(c_o2 @ vj_o1 @ c_o2)
        self.ground_offset = float(
            ref.fock_beta_mo[ref.o1, ref.o1]
            - ref.fock_alpha_mo[ref.o2, ref.o2]
            - self.lr.exact_exchange * eri_o2o2_o1o1
        )
        self.ground_offset_source_error = (
            self.ground_offset_operator - self.ground_offset
        )

    def build_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(A_EMRSF, live_MRSF_indices, C)``."""

        a_mrsf_full = self.mrsf.build_matrix()
        live = self.live
        a_mrsf = a_mrsf_full[np.ix_(live, live)]

        a_cv = self.lr.build_matrix() + np.eye(self.lr.space.size) * self.ground_offset

        eye = np.eye(self.lr.space.size)
        c_full = np.column_stack([self.coupling.matvec(eye[:, k]) for k in range(self.lr.space.size)]) if self.lr.space.size else np.zeros((self.mrsf.space.size, 0))
        coupling = c_full[live, :]
        matrix = np.block([[a_mrsf, coupling], [coupling.T, a_cv]])
        matrix = 0.5 * (matrix + matrix.T)
        return matrix, live, coupling

    @property
    def dimension(self) -> int:
        return len(self.live) + self.lr.space.size

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.dimension:
            raise ValueError(f"Expected an EMRSF vector of length {self.dimension}")
        nm = len(self.live)
        xm, xcv = vector[:nm], vector[nm:]
        full_m = np.zeros(self.mrsf.space.size)
        full_m[self.live] = xm

        c_x, ct_x = self.coupling.apply(xcv, full_m)
        ym = self.mrsf.matvec(full_m)
        ym += c_x
        ycv = self.lr.matvec(xcv) + self.ground_offset * xcv
        ycv += ct_x
        return np.concatenate((ym[self.live], ycv))

    def diagonal_guess(self) -> np.ndarray:
        mrsf = self.mrsf.space.diagonal_guess()[self.live]
        cv = self.lr.diagonal_guess() + self.ground_offset
        return np.concatenate((mrsf, cv))

    def resource_estimate(self):
        """Estimate the dominant dense, Davidson and integral storage."""

        return estimate_emrsf_resources(self)

    def _state_decompositions(
        self, values: np.ndarray, vectors: np.ndarray
    ) -> tuple[EMRSFStateDecomposition, ...]:
        """Resolve every converged eigenvalue into the three EMRSF blocks."""

        nm = len(self.live)
        decompositions = []
        for state, eigenvalue in enumerate(values):
            xm = np.asarray(vectors[:nm, state], dtype=float)
            xcv = np.asarray(vectors[nm:, state], dtype=float)
            full_m = np.zeros(self.mrsf.space.size)
            full_m[self.live] = xm

            am = self.mrsf.matvec(full_m)[self.live]
            acv = self.lr.matvec(xcv) + self.ground_offset * xcv
            coupling = self.coupling.matvec(xcv)[self.live]

            mrsf_value = float(np.dot(xm, am))
            cv_value = float(np.dot(xcv, acv))
            coupling_value = float(2.0 * np.dot(xm, coupling))
            reconstructed = mrsf_value + cv_value + coupling_value
            decompositions.append(
                EMRSFStateDecomposition(
                    state=state,
                    eigenvalue_hartree=float(eigenvalue),
                    mrsf_weight=float(np.dot(xm, xm)),
                    cv_weight=float(np.dot(xcv, xcv)),
                    mrsf_block_hartree=mrsf_value,
                    cv_block_hartree=cv_value,
                    coupling_hartree=coupling_value,
                    reconstructed_hartree=reconstructed,
                    closure_error_hartree=float(reconstructed - eigenvalue),
                )
            )
        return tuple(decompositions)

    def kernel(self, *, solver: str | None = None) -> EMRSFResult:
        solver = self.solver if solver is None else solver
        mrsf_result = None
        if self.solve_mrsf_first:
            if self.progress:
                print(
                    "[EMRSF] solving optional standalone MRSF diagnostics",
                    flush=True,
                )
            mrsf_result = self.mrsf.kernel(solver=solver)
            if not np.all(mrsf_result.converged):
                raise RuntimeError(
                    "MRSF iterative diagonalisation did not converge all roots; "
                    f"residuals={mrsf_result.residual_norms}"
                )
        elif self.progress:
            print(
                "[EMRSF] direct enlarged-space mode: MRSF pre-diagonalisation skipped",
                flush=True,
            )
        if solver.lower() != "dense":
            if self.progress:
                print("[EMRSF] solving the coupled matrix-free EMRSF operator", flush=True)
            from .symmetry import cv_basis_irrep_ids, mrsf_basis_irrep_ids

            mrsf_irreps, mrsf_group = mrsf_basis_irrep_ids(self.mrsf.space)
            cv_irreps, cv_group = cv_basis_irrep_ids(self.lr.space)
            if mrsf_group != cv_group:
                raise ValueError("MRSF and LR sectors use different point groups")
            basis_irreps = np.concatenate((mrsf_irreps[self.live], cv_irreps))
            iterative = solve_lowest(
                self.matvec,
                self.diagonal_guess(),
                solver=solver,
                nroots=min(self.nstates, self.dimension),
                tol=self.conv_tol,
                residual_tol=self.residual_tol,
                max_cycle=self.max_cycle,
                max_space=self.max_space,
                max_memory=getattr(self.mrsf.ref.mf, "max_memory", 4000),
                verbose=max(0, getattr(self.mrsf.ref.mf, "verbose", 0) - 2),
                progress=self.progress,
                progress_label="EMRSF Davidson",
                basis_irreps=basis_irreps,
            )
            decompositions = self._state_decompositions(
                iterative.eigenvalues, iterative.eigenvectors
            )
            return EMRSFResult(
                eigenvalues=iterative.eigenvalues,
                eigenvectors=iterative.eigenvectors,
                matrix=None,
                mrsf_size=len(self.live),
                cv_space=self.lr.space,
                mrsf_result=mrsf_result,
                live_mrsf_indices=self.live,
                converged=iterative.converged,
                residual_norms=iterative.residual_norms,
                iterations=iterative.iterations,
                matvecs=iterative.matvecs,
                solver=iterative.solver,
                initial_guesses=iterative.initial_guesses,
                symmetry_seed_counts=iterative.symmetry_seed_counts,
                exact_exchange=self.lr.exact_exchange,
                coupling_scale=self.coupling_scale,
                ground_offset_hartree=self.ground_offset,
                ground_offset_operator_hartree=self.ground_offset_operator,
                ground_offset_source_error_hartree=self.ground_offset_source_error,
                mrsf_space_object=self.mrsf.space,
                direct_emrsf=not self.solve_mrsf_first,
                integral_backend=self.integral_backend,
                state_decompositions=decompositions,
            )

        matrix, live, coupling = self.build_matrix()
        if matrix.size == 0:
            raise ValueError("The EMRSF response space is empty")
        nroots = min(self.nstates, len(matrix))
        values, vectors = eigh(matrix, subset_by_index=(0, nroots - 1))
        decompositions = self._state_decompositions(values, vectors)
        return EMRSFResult(
            eigenvalues=values,
            eigenvectors=vectors,
            matrix=matrix,
            mrsf_size=len(live),
            cv_space=self.lr.space,
            mrsf_result=mrsf_result,
            live_mrsf_indices=live,
            converged=np.ones(len(values), dtype=bool),
            residual_norms=np.zeros(len(values)),
            solver="dense",
            exact_exchange=self.lr.exact_exchange,
            coupling_scale=self.coupling_scale,
            ground_offset_hartree=self.ground_offset,
            ground_offset_operator_hartree=self.ground_offset_operator,
            ground_offset_source_error_hartree=self.ground_offset_source_error,
            coupling_frobenius_norm=float(np.linalg.norm(coupling)),
            mrsf_space_object=self.mrsf.space,
            direct_emrsf=not self.solve_mrsf_first,
            integral_backend=self.integral_backend,
            state_decompositions=decompositions,
        )
