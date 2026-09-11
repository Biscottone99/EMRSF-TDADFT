from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyscf_emrsf import MRSFTDA
from pyscf_emrsf.postprocess import save_postprocessing_archive
from pyscf_emrsf.properties import mrsf_transition_property
from pyscf_emrsf.space import MRSFSpace


class _IntegralMolecule:
    def __init__(self, nmo: int):
        self.nmo = nmo

    def intor_symmetric(self, name, comp=None):
        if name == "int1e_ovlp":
            return np.eye(self.nmo)
        if name == "int1e_r" and comp == 3:
            values = np.zeros((3, self.nmo, self.nmo))
            values[0, 1, 3] = values[0, 3, 1] = 0.5
            return values
        raise AssertionError((name, comp))


class _SmallReference:
    def __init__(self):
        self.closed = np.array([0])
        self.open = np.array([1, 2])
        self.virtual = np.array([3])
        self.occupation = np.array([2.0, 1.0, 1.0, 0.0])
        self.coeff = np.eye(4)
        self.mf = SimpleNamespace(mol=_IntegralMolecule(4))

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


def test_length_gauge_oscillator_strength_for_pure_pair():
    space = MRSFSpace(_SmallReference(), target_multiplicity=1)
    ground = _unit_vector(space, (space.ref.o2, space.ref.o1))
    excited = _unit_vector(space, (space.ref.o2, space.ref.virtual[0]))
    result = SimpleNamespace(
        space=space,
        eigenvectors=np.column_stack((ground, excited)),
        eigenvalues=np.array([0.0, 1.0]),
    )

    value = mrsf_transition_property(result, 1)
    np.testing.assert_allclose(
        value.transition_dipole_e_bohr,
        [-1.0 / np.sqrt(2.0), 0.0, 0.0],
    )
    assert np.isclose(value.oscillator_strength, 1.0 / 3.0)
    assert abs(value.transition_density_trace) < 1.0e-14


def test_postprocessing_archive_is_pickle_free_and_complete(h2o_rohf, tmp_path):
    result = MRSFTDA(
        h2o_rohf,
        nstates=2,
        density_fit=False,
        solver="dense",
    ).kernel()
    properties = result.transition_properties()
    export = save_postprocessing_archive(
        result,
        tmp_path / "mrsf_postprocessing.npz",
        transition_properties=properties,
    )

    assert export.archive.is_file()
    assert export.manifest.is_file()
    with np.load(export.archive, allow_pickle=False) as archive:
        np.testing.assert_allclose(
            archive["eigenvectors_columns_are_states"], result.eigenvectors
        )
        assert archive["mrsf_labels_zero_based"].shape[1] == 2
        assert archive["mo_coeff_ao_by_mo"].shape[0] == h2o_rohf.mol.nao_nr()
        assert archive["ao_position_integrals_bohr"].shape[0] == 3
        assert "oscillator_strength_length" in str(
            archive["mrsf_transition_properties_json"]
        )
