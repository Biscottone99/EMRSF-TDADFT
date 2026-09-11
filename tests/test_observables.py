from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyscf_emrsf.observables import (
    BOHR_TO_ANGSTROM,
    ChargeTransferDescriptor,
    IntegrationGrid,
    StateDensity,
    charge_transfer_descriptor,
    mrsf_state_density,
    mrsf_transition_difference_density_mo,
)
from pyscf_emrsf.space import MRSFSpace
from pyscf_emrsf.symmetry import classify_vector


class _SmallReference:
    def __init__(self):
        self.closed = np.array([0])
        self.open = np.array([1, 2])
        self.virtual = np.array([3])
        self.occupation = np.array([2.0, 1.0, 1.0, 0.0])
        self.coeff = np.eye(4)

    @property
    def nao(self):
        return 4

    @property
    def nmo(self):
        return 4

    @property
    def o1(self):
        return 1

    @property
    def o2(self):
        return 2


def _unit_vector(space, pair):
    vector = np.zeros(space.size)
    vector[space.labels.index(pair)] = 1.0
    return vector


def test_pure_configuration_difference_densities():
    space = MRSFSpace(_SmallReference(), target_multiplicity=1)

    ground = _unit_vector(space, (space.ref.o2, space.ref.o1))
    density = mrsf_transition_difference_density_mo(space, ground, ground)
    np.testing.assert_allclose(np.diag(density), [0.0, 1.0, -1.0, 0.0])

    cv = _unit_vector(space, (0, 3))
    density = mrsf_transition_difference_density_mo(space, cv, cv)
    np.testing.assert_allclose(np.diag(density), [-1.0, 0.0, 0.0, 1.0])


def test_oos_expansion_has_triplet_spin_summed_density():
    space = MRSFSpace(_SmallReference(), target_multiplicity=1)
    oos = _unit_vector(space, (space.ref.o1, space.ref.o1))
    density = mrsf_transition_difference_density_mo(space, oos, oos)
    np.testing.assert_allclose(density, np.zeros((4, 4)), atol=1.0e-15)


def test_state_density_particle_number_is_conserved():
    space = MRSFSpace(_SmallReference(), target_multiplicity=1)
    rng = np.random.default_rng(91)
    vector = space.mask_redundant(rng.normal(size=space.size))
    vector /= np.linalg.norm(vector)
    result = SimpleNamespace(space=space, eigenvectors=vector[:, None])
    density = mrsf_state_density(result, 0)

    assert abs(density.particle_change_mo) < 1.0e-13
    np.testing.assert_allclose(np.trace(density.state_mo), 4.0, atol=1.0e-13)
    np.testing.assert_allclose(density.difference_mo, density.difference_mo.T)


def test_descriptor_unit_conversions():
    descriptor = ChargeTransferDescriptor(
        state_index=0,
        q_plus=0.5,
        q_minus=0.5,
        q=0.5,
        r_plus_bohr=np.array([2.0, 0.0, 0.0]),
        r_minus_bohr=np.zeros(3),
        d_ct_bohr=2.0,
        mu_e_bohr=1.0,
        integral_delta_rho=0.0,
        reference_electrons_grid=4.0,
        state_electrons_grid=4.0,
        grid_level=5,
        grid_points=100,
        grid_pruned=True,
    )

    assert descriptor.d_ct_angstrom == 2.0 * BOHR_TO_ANGSTROM
    assert descriptor.mu_e_angstrom == BOHR_TO_ANGSTROM
    assert descriptor.mu_debye > 2.5


def test_response_vector_symmetry_classification():
    vector = np.array([0.0, np.sqrt(0.9), np.sqrt(0.1), 0.0])
    basis_irreps = np.array([0, 2, 3, 3])
    value = classify_vector(vector, basis_irreps, "C2v")

    assert value.irrep_id == 2
    assert np.isclose(value.purity, 0.9)


def test_s45_uses_the_common_s44_charge(monkeypatch):
    # Deliberately imbalanced finite quadrature: q+=2, q-=1 and q=1.5.
    # The SI S45 centroids must both use the common q denominator.
    class _NumInt:
        @staticmethod
        def eval_ao(_mol, coordinates, deriv=0):
            return np.eye(2)

    import pyscf.dft.numint as numint

    monkeypatch.setattr(numint, "eval_ao", _NumInt.eval_ao)
    density = StateDensity(
        state_index=0,
        reference_mo=np.zeros((2, 2)),
        difference_mo=np.zeros((2, 2)),
        state_mo=np.zeros((2, 2)),
        reference_ao=np.zeros((2, 2)),
        difference_ao=np.diag([1.0, -1.0]),
        state_ao=np.diag([1.0, -1.0]),
    )
    grid = IntegrationGrid(
        coordinates=np.array([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        weights=np.array([2.0, 1.0]),
        level=1,
        pruned=False,
    )
    value = charge_transfer_descriptor(density, object(), grid)
    np.testing.assert_allclose(value.r_plus_bohr, [4.0 / 1.5, 0.0, 0.0])
    np.testing.assert_allclose(value.r_minus_bohr, [-1.0 / 1.5, 0.0, 0.0])
