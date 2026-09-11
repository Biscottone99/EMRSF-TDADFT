from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyscf_emrsf.observables import mrsf_transition_difference_density_mo
from pyscf_emrsf.orbitals import mrsf_natural_transition_orbitals
from pyscf_emrsf.space import MRSFSpace


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


def test_mrsf_nto_recovers_a_pure_hole_particle_pair():
    space = MRSFSpace(_SmallReference(), target_multiplicity=1)
    ground = _unit_vector(space, (space.ref.o2, space.ref.o1))
    excited = _unit_vector(space, (space.ref.o2, space.ref.virtual[0]))
    result = SimpleNamespace(
        space=space,
        eigenvectors=np.column_stack((ground, excited)),
        eigenvalues=np.array([0.0, 1.0]),
    )

    transition = mrsf_transition_difference_density_mo(space, ground, excited)
    assert np.isclose(transition[space.ref.virtual[0], space.ref.o1], np.sqrt(2.0))

    nto = mrsf_natural_transition_orbitals(result, 1)
    assert np.isclose(nto.transition_norm, np.sqrt(2.0))
    assert np.isclose(nto.weights[0], 1.0)
    np.testing.assert_allclose(
        np.abs(nto.hole_mo[:, 0]),
        np.eye(space.ref.nmo)[:, space.ref.o1],
    )
    np.testing.assert_allclose(
        np.abs(nto.particle_mo[:, 0]),
        np.eye(space.ref.nmo)[:, space.ref.virtual[0]],
    )


def test_reference_and_nto_molden_exports(h2o_rohf, tmp_path):
    from pyscf_emrsf import EMRSFTDA

    result = EMRSFTDA(h2o_rohf, nstates=3).kernel()
    orbital_file = result.write_reference_molden(tmp_path / "orbitals.molden")
    assert orbital_file.is_file()
    assert "[MO]" in orbital_file.read_text(encoding="utf-8")

    export = result.write_mrsf_nto_molden(
        tmp_path / "nto",
        final_states=[1],
        weight_threshold=1.0e-6,
    )
    assert export.summary_csv.is_file()
    assert len(export.molden_files) == 1
    assert "[MO]" in export.molden_files[0].read_text(encoding="utf-8")
    summary = export.summary_csv.read_text(encoding="utf-8")
    assert "transition_density_norm" in summary
    assert "WRITTEN" in summary
