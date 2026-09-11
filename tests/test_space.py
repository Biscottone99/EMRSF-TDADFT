from __future__ import annotations

import numpy as np

from pyscf_emrsf.reference import analyse_reference
from pyscf_emrsf.space import MRSFSpace


def test_pack_roundtrip_and_labels(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    space = MRSFSpace(ref, 1)
    vector = np.arange(space.size, dtype=float)
    assert np.allclose(space.pack(space.unpack(vector)), vector)
    assert space.size == len(space.labels)


def test_triplet_projection_masks_open_open_slots(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    space = MRSFSpace(ref, 3)
    x = np.ones(space.size)
    projected = space.unpack(space.mask_redundant(x))
    assert projected[ref.o2, ref.o2] == 0.0
    assert projected[ref.o1, ref.o2] == 0.0
    assert projected[ref.o2, ref.o1] == 0.0


def test_ao_components_are_linear(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    space = MRSFSpace(ref, 1)
    rng = np.random.default_rng(7)
    x = rng.normal(size=space.size)
    y = rng.normal(size=space.size)
    assert np.allclose(
        space.ao_components(0.3 * x - 0.8 * y),
        0.3 * space.ao_components(x) - 0.8 * space.ao_components(y),
    )

