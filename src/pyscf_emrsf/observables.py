"""State densities and charge-transfer descriptors for MRSF response states.

The scalar descriptors implement eqs. S42--S47 of Oh *et al.*.  The MRSF
one-particle density uses the spin-adapted OOS transformation employed by the
reference implementation: the stored OOS amplitude is expanded as
``O1->O1 / sqrt(2) - O2->O2 / sqrt(2)`` for a singlet before the occupied and
beta-unoccupied density blocks are contracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
E_ANGSTROM_TO_DEBYE = 4.803204712570263


@dataclass(frozen=True)
class StateDensity:
    """Spin-summed, unrelaxed one-particle densities for one MRSF state."""

    state_index: int
    reference_mo: np.ndarray
    difference_mo: np.ndarray
    state_mo: np.ndarray
    reference_ao: np.ndarray
    difference_ao: np.ndarray
    state_ao: np.ndarray

    @property
    def particle_change_mo(self) -> float:
        """Trace of the difference 1-RDM; it should be zero."""

        return float(np.trace(self.difference_mo))

    def fields(self, mol, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``(delta_rho, rho_plus, rho_minus)`` at Cartesian points.

        Coordinates are in bohr, as required by PySCF AO evaluation.
        """

        from pyscf.dft import numint

        ao = numint.eval_ao(mol, np.asarray(coordinates, dtype=float), deriv=0)
        delta = np.einsum(
            "pi,ij,pj->p", ao, self.difference_ao, ao, optimize=True
        )
        return delta, np.maximum(delta, 0.0), np.maximum(-delta, 0.0)


@dataclass(frozen=True)
class IntegrationGrid:
    """Atom-centred numerical quadrature used for eqs. S43--S47."""

    coordinates: np.ndarray
    weights: np.ndarray
    level: int
    pruned: bool

    @property
    def size(self) -> int:
        return int(len(self.weights))


@dataclass(frozen=True)
class ChargeTransferDescriptor:
    """Numerical realization of the density-difference observables S42--S47."""

    state_index: int
    q_plus: float
    q_minus: float
    q: float
    r_plus_bohr: np.ndarray
    r_minus_bohr: np.ndarray
    d_ct_bohr: float
    mu_e_bohr: float
    integral_delta_rho: float
    reference_electrons_grid: float
    state_electrons_grid: float
    grid_level: int
    grid_points: int
    grid_pruned: bool

    @property
    def charge_imbalance(self) -> float:
        return float(self.q_plus - self.q_minus)

    @property
    def r_plus_angstrom(self) -> np.ndarray:
        return np.asarray(self.r_plus_bohr) * BOHR_TO_ANGSTROM

    @property
    def r_minus_angstrom(self) -> np.ndarray:
        return np.asarray(self.r_minus_bohr) * BOHR_TO_ANGSTROM

    @property
    def d_ct_angstrom(self) -> float:
        return float(self.d_ct_bohr * BOHR_TO_ANGSTROM)

    @property
    def mu_e_angstrom(self) -> float:
        """Paper-compatible ``q * D_CT`` in electron-angstrom."""

        return float(self.q * self.d_ct_angstrom)

    @property
    def mu_debye(self) -> float:
        return float(self.mu_e_angstrom * E_ANGSTROM_TO_DEBYE)

    def as_dict(self) -> dict[str, object]:
        return {
            "state_index": self.state_index,
            "q_plus_e": self.q_plus,
            "q_minus_e": self.q_minus,
            "q_e": self.q,
            "r_plus_bohr": self.r_plus_bohr.tolist(),
            "r_minus_bohr": self.r_minus_bohr.tolist(),
            "r_plus_angstrom": self.r_plus_angstrom.tolist(),
            "r_minus_angstrom": self.r_minus_angstrom.tolist(),
            "d_ct_bohr": self.d_ct_bohr,
            "d_ct_angstrom": self.d_ct_angstrom,
            "mu_e_bohr": self.mu_e_bohr,
            "mu_e_angstrom": self.mu_e_angstrom,
            "mu_debye": self.mu_debye,
            "integral_delta_rho": self.integral_delta_rho,
            "charge_imbalance": self.charge_imbalance,
            "reference_electrons_grid": self.reference_electrons_grid,
            "state_electrons_grid": self.state_electrons_grid,
            "grid_level": self.grid_level,
            "grid_points": self.grid_points,
            "grid_pruned": self.grid_pruned,
        }


def _expanded_amplitudes(space, vector: np.ndarray) -> np.ndarray:
    """Return the raw alpha-occupied/beta-unoccupied amplitude matrix."""

    vector = space.mask_redundant(vector)
    raw = space.unpack(vector)
    o1, o2 = space.ref.o1, space.ref.o2
    amplitude = raw[o1, o1]
    raw[o1, o1] = amplitude / np.sqrt(2.0)
    if space.target_multiplicity == 1:
        raw[o2, o2] = -amplitude / np.sqrt(2.0)
    else:
        raw[o2, o2] = amplitude / np.sqrt(2.0)
    return raw


def mrsf_transition_difference_density_mo(
    space, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Build an unrelaxed MRSF transition/difference density in the MO basis.

    For ``left is right`` this is the state-density difference relative to
    the high-spin triplet reference.  The square-root-of-two factors are the
    dimensional-transformation corrections for contractions in which exactly
    one of the two amplitudes belongs to the O1/O2 open--open subspace.
    """

    xl = _expanded_amplitudes(space, np.asarray(left, dtype=float))
    xr = _expanded_amplitudes(space, np.asarray(right, dtype=float))
    rows = np.asarray(space.rows, dtype=int)
    cols = np.asarray(space.cols, dtype=int)
    opened = {space.ref.o1, space.ref.o2}
    density = np.zeros((space.ref.nmo, space.ref.nmo))
    sqrt2 = np.sqrt(2.0)

    for i in rows:
        for a in cols:
            for b in cols:
                special_a = i in opened and a in opened
                special_b = i in opened and b in opened
                scale = sqrt2 if special_a != special_b else 1.0
                density[a, b] += scale * xr[i, a] * xl[i, b]

    for i in rows:
        for j in rows:
            for a in cols:
                special_i = i in opened and a in opened
                special_j = j in opened and a in opened
                scale = sqrt2 if special_i != special_j else 1.0
                density[i, j] -= scale * xr[j, a] * xl[i, a]

    return density


def mrsf_state_density(result, state: int) -> StateDensity:
    """Return triplet-reference, difference, and state 1-RDMs for one root."""

    state = int(state)
    if state < 0 or state >= result.eigenvectors.shape[1]:
        raise IndexError(f"State {state} is outside the computed root range")
    vector = result.eigenvectors[:, state]
    difference_mo = mrsf_transition_difference_density_mo(
        result.space, vector, vector
    )
    difference_mo = 0.5 * (difference_mo + difference_mo.T)
    reference_mo = np.diag(np.asarray(result.space.ref.occupation, dtype=float))
    state_mo = reference_mo + difference_mo
    coeff = result.space.ref.coeff
    reference_ao = coeff @ reference_mo @ coeff.T
    difference_ao = coeff @ difference_mo @ coeff.T
    state_ao = coeff @ state_mo @ coeff.T
    return StateDensity(
        state_index=state,
        reference_mo=reference_mo,
        difference_mo=difference_mo,
        state_mo=state_mo,
        reference_ao=reference_ao,
        difference_ao=difference_ao,
        state_ao=state_ao,
    )


def build_integration_grid(mol, *, level: int = 5, prune: bool = True) -> IntegrationGrid:
    """Build a deterministic PySCF molecular DFT grid in atomic units."""

    from pyscf.dft import gen_grid

    grid = gen_grid.Grids(mol)
    grid.level = int(level)
    if not prune:
        grid.prune = None
    grid.build(with_non0tab=False)
    return IntegrationGrid(
        coordinates=np.asarray(grid.coords),
        weights=np.asarray(grid.weights),
        level=int(level),
        pruned=bool(prune),
    )


def charge_transfer_descriptor(
    state_density: StateDensity,
    mol,
    grid: IntegrationGrid,
    *,
    block_size: int = 20000,
    zero_threshold: float = 1.0e-12,
) -> ChargeTransferDescriptor:
    """Integrate eqs. S42--S47 for one state on a supplied grid."""

    from pyscf.dft import numint

    q_plus = 0.0
    q_minus = 0.0
    moment_plus = np.zeros(3)
    moment_minus = np.zeros(3)
    integral_delta = 0.0
    reference_electrons = 0.0
    state_electrons = 0.0
    npoint = grid.size
    block_size = max(1, int(block_size))

    for start in range(0, npoint, block_size):
        stop = min(start + block_size, npoint)
        coordinates = grid.coordinates[start:stop]
        weights = grid.weights[start:stop]
        ao = numint.eval_ao(mol, coordinates, deriv=0)
        delta = np.einsum(
            "pi,ij,pj->p", ao, state_density.difference_ao, ao, optimize=True
        )
        rho_ref = np.einsum(
            "pi,ij,pj->p", ao, state_density.reference_ao, ao, optimize=True
        )
        rho_state = rho_ref + delta
        positive = np.maximum(delta, 0.0)
        negative = np.maximum(-delta, 0.0)
        wp = weights * positive
        wm = weights * negative

        q_plus += float(np.sum(wp))
        q_minus += float(np.sum(wm))
        moment_plus += np.einsum("p,px->x", wp, coordinates, optimize=True)
        moment_minus += np.einsum("p,px->x", wm, coordinates, optimize=True)
        integral_delta += float(np.dot(weights, delta))
        reference_electrons += float(np.dot(weights, rho_ref))
        state_electrons += float(np.dot(weights, rho_state))

    # Eq. S44 equates the two integrals.  Their symmetric average is the most
    # stable finite-grid estimate; the separate values and residual are kept
    # as explicit quadrature diagnostics in the result and log.
    q = 0.5 * (q_plus + q_minus)
    if q <= zero_threshold:
        r_plus = np.full(3, np.nan)
        r_minus = np.full(3, np.nan)
        d_ct = 0.0
    else:
        # Eq. S45 uses the *same* transferred charge q from S44 for both
        # centroids.  q+ and q- remain available as finite-grid diagnostics,
        # but substituting separate denominators changes the published
        # observable whenever the quadrature has a small charge imbalance.
        r_plus = moment_plus / q
        r_minus = moment_minus / q
        d_ct = float(np.linalg.norm(r_plus - r_minus))

    return ChargeTransferDescriptor(
        state_index=state_density.state_index,
        q_plus=q_plus,
        q_minus=q_minus,
        q=q,
        r_plus_bohr=r_plus,
        r_minus_bohr=r_minus,
        d_ct_bohr=d_ct,
        mu_e_bohr=float(q * d_ct),
        integral_delta_rho=integral_delta,
        reference_electrons_grid=reference_electrons,
        state_electrons_grid=state_electrons,
        grid_level=grid.level,
        grid_points=grid.size,
        grid_pruned=grid.pruned,
    )


def analyse_mrsf_charge_transfer(
    result,
    *,
    states: Iterable[int] | None = None,
    grid_level: int = 5,
    prune: bool = True,
    block_size: int = 20000,
) -> tuple[ChargeTransferDescriptor, ...]:
    """Compute S42--S47 for several MRSF roots using one shared grid."""

    if states is None:
        states = range(result.eigenvectors.shape[1])
    selected = tuple(int(state) for state in states)
    grid = build_integration_grid(
        result.space.ref.mf.mol, level=grid_level, prune=prune
    )
    return tuple(
        charge_transfer_descriptor(
            mrsf_state_density(result, state),
            result.space.ref.mf.mol,
            grid,
            block_size=block_size,
        )
        for state in selected
    )


def density_grid_fields(
    result,
    state: int,
    *,
    grid_level: int = 5,
    prune: bool = True,
    block_size: int = 20000,
) -> dict[str, np.ndarray | int | bool]:
    """Materialize the S42/S43 fields for validation or compressed export."""

    density = mrsf_state_density(result, state)
    grid = build_integration_grid(
        result.space.ref.mf.mol, level=grid_level, prune=prune
    )
    delta = np.empty(grid.size)
    positive = np.empty(grid.size)
    negative = np.empty(grid.size)
    block_size = max(1, int(block_size))
    for start in range(0, grid.size, block_size):
        stop = min(start + block_size, grid.size)
        values = density.fields(result.space.ref.mf.mol, grid.coordinates[start:stop])
        delta[start:stop], positive[start:stop], negative[start:stop] = values
    return {
        "coordinates_bohr": grid.coordinates,
        "weights_bohr3": grid.weights,
        "delta_rho": delta,
        "rho_plus": positive,
        "rho_minus": negative,
        "grid_level": grid.level,
        "grid_pruned": grid.pruned,
    }
