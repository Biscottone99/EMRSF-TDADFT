"""Two-electron response contractions for MRSF-TDA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import scf


@dataclass(frozen=True)
class ExchangeModel:
    """Short- and long-range exact-exchange fractions."""

    short_range: float
    long_range: float
    omega: float


def exchange_model(mf) -> ExchangeModel:
    if not hasattr(mf, "xc"):
        return ExchangeModel(1.0, 1.0, 0.0)
    from pyscf import dft

    omega, alpha, beta = dft.libxc.rsh_coeff(mf.xc)
    # PySCF convention: alpha is the long-range fraction and alpha+beta the
    # short-range fraction (beta can therefore be negative for CAM-B3LYP).
    return ExchangeModel(float(alpha + beta), float(alpha), float(omega))


def _jk(mol, density: np.ndarray, omega: float | None = None):
    if omega is None or abs(omega) < 1.0e-15:
        return scf.hf.get_jk(mol, density, hermi=0)
    with mol.with_range_coulomb(omega):
        return scf.hf.get_jk(mol, density, hermi=0)


def mrsf_two_electron_components(
    mf,
    density_components: np.ndarray,
    *,
    target_multiplicity: int,
    spc_coco: float | None = None,
    spc_ovov: float | None = None,
    spc_coov: float | None = None,
    jk=None,
) -> np.ndarray:
    """Build seven AO Fock-like MRSF response components.

    This is a direct PySCF ``get_jk`` transcription of the MRSF-TDA
    Coulomb/exchange contractions.  It deliberately mirrors the component
    algebra instead of importing or calling OpenQP.

    Notes
    -----
    MRSF-TDDFT uses the exact-exchange channel of a hybrid functional for the
    spin-pair terms; semilocal XC-kernel contributions are absent in this
    collinear spin-flip channel.  Range separation is handled by combining
    full-range and long-range exchange operators with PySCF/Libcint.
    """

    d = np.asarray(density_components)
    if d.ndim != 3 or d.shape[0] != 7:
        raise ValueError("density_components must have shape (7, nao, nao)")
    model = exchange_model(mf)
    out = np.zeros_like(d)

    # The OpenQP MRSF contraction scales Coulomb and exchange together by the
    # short-range HF fraction, then adds the difference to the general exchange
    # component for CAM/LRC functionals.
    if jk is None:
        vj, vk = _jk(mf.mol, d)
    else:
        vj, vk = jk.get_jk(d, hermi=0)
    out[:4] += model.short_range * vj[:4]
    out -= model.short_range * vk

    if model.omega > 0.0 and abs(model.long_range - model.short_range) > 1.0e-15:
        if jk is None:
            _, vk_lr = _jk(mf.mol, d[6], model.omega)
        else:
            _, vk_lr = jk.get_jk(d[6], hermi=0, omega=model.omega)
        out[6] -= (model.long_range - model.short_range) * vk_lr

    if target_multiplicity == 3:
        out[:6] *= -1.0

    base = model.short_range
    default = base
    spc_coco = default if spc_coco is None else spc_coco
    spc_ovov = default if spc_ovov is None else spc_ovov
    spc_coov = default if spc_coov is None else spc_coov
    if abs(base) < 1.0e-15:
        if any(abs(x) > 1.0e-15 for x in (spc_coco, spc_ovov, spc_coov)):
            raise ValueError("Nonzero spin-pair coupling requires exact exchange")
    else:
        out[5] *= spc_coco / base
        out[4] *= spc_ovov / base
        out[:4] *= spc_coov / base
    return out
