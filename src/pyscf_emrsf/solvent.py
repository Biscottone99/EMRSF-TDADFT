"""Reference-state polarizable-continuum solvation for MRSF/EMRSF.

The solvent is relaxed self-consistently with the high-spin ROKS/ROHF
reference.  The resulting static reaction potential is retained in every
spin-free MRSF/EMRSF one-electron block.  No unpublished excited-state PCM
response kernel is added to the MRSF equations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


_PCM_METHODS = {
    "c-pcm": "C-PCM",
    "cpcm": "C-PCM",
    "cosmo": "COSMO",
    "ief-pcm": "IEF-PCM",
    "iefpcm": "IEF-PCM",
    "ss(v)pe": "SS(V)PE",
    "ssvpe": "SS(V)PE",
}


@dataclass(frozen=True)
class PCMConfig:
    """Settings for self-consistent reference-state PySCF PCM."""

    epsilon: float
    method: str = "IEF-PCM"
    lebedev_order: int = 29
    vdw_scale: float = 1.2
    probe_radius_angstrom: float = 0.0
    surface_method: str = "SWIG"
    max_cycle: int = 50
    conv_tol: float = 1.0e-9

    def __post_init__(self) -> None:
        canonical = _PCM_METHODS.get(str(self.method).strip().lower())
        if canonical is None:
            raise ValueError(
                "PCM method must be C-PCM, COSMO, IEF-PCM, or SS(V)PE"
            )
        object.__setattr__(self, "method", canonical)
        surface = str(self.surface_method).strip().upper()
        if surface not in {"SWIG", "ISWIG"}:
            raise ValueError("PCM surface_method must be SWIG or ISWIG")
        object.__setattr__(self, "surface_method", surface)
        if float(self.epsilon) <= 1.0:
            raise ValueError("PCM epsilon must be greater than 1")
        if int(self.lebedev_order) < 3:
            raise ValueError("PCM lebedev_order is not a valid positive order")
        if float(self.vdw_scale) <= 0.0:
            raise ValueError("PCM vdw_scale must be positive")
        if float(self.probe_radius_angstrom) < 0.0:
            raise ValueError("PCM probe radius must be non-negative")
        if int(self.max_cycle) < 1 or float(self.conv_tol) <= 0.0:
            raise ValueError("PCM convergence controls must be positive")

    def as_metadata(self) -> dict[str, object]:
        values = asdict(self)
        values["scope"] = "self-consistent reference; frozen in response"
        values["excited_state_reaction_field"] = False
        return values


def attach_pcm(mf, config: PCMConfig):
    """Return ``mf`` wrapped in a freshly configured PySCF PCM object."""

    from pyscf import solvent
    from pyscf.dft import gen_grid

    if config.lebedev_order not in gen_grid.LEBEDEV_ORDER:
        available = ", ".join(map(str, sorted(gen_grid.LEBEDEV_ORDER)))
        raise ValueError(
            f"Unsupported PCM Lebedev order {config.lebedev_order}; "
            f"available orders are: {available}"
        )

    pcm = solvent.PCM(mf.mol)
    pcm.method = config.method
    pcm.eps = float(config.epsilon)
    pcm.lebedev_order = int(config.lebedev_order)
    pcm.vdw_scale = float(config.vdw_scale)
    pcm.r_probe = float(config.probe_radius_angstrom)
    pcm.surface_discretization_method = config.surface_method
    pcm.max_cycle = int(config.max_cycle)
    pcm.conv_tol = float(config.conv_tol)
    pcm.equilibrium_solvation = False
    wrapped = solvent.PCM(mf, pcm)
    wrapped.emrsf_pcm_config = config
    return wrapped


def pcm_metadata(mf) -> dict[str, object] | None:
    """Return serializable PCM settings attached to a mean-field object."""

    config = getattr(mf, "emrsf_pcm_config", None)
    return None if config is None else config.as_metadata()
