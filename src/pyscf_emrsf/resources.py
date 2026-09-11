"""Transparent memory estimates for dense and iterative EMRSF calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceEstimate:
    response_dimension: int
    dense_matrix_gib: float
    davidson_vectors_gib: float
    ri_coupling_intermediates_gib: float | None
    direct_coupling_intermediates_gib: float | None
    ao_three_index_gib: float | None
    integral_backend: str


def estimate_emrsf_resources(method) -> ResourceEstimate:
    nm = len(method.live)
    nc = len(method.mrsf.ref.closed)
    nv = len(method.mrsf.ref.virtual)
    dimension = nm + nc * nv
    nroots = min(method.nstates, dimension)
    max_space = method.max_space
    if max_space is None:
        max_space = min(dimension, max(40, 8 * nroots + 12))

    gib = 1024.0**3
    dense = 8.0 * dimension**2 / gib
    # PySCF Davidson holds basis/Abasis plus current Ritz and residual vectors.
    davidson = 8.0 * dimension * (2 * max_space + 3 * nroots) / gib
    # Seven Cx/adjoint correction tables plus the nc^2 and nv^2 products.
    ri_coupling = 8.0 * (
        7 * nc * nv + nc * nc + nv * nv + nc + nv
    ) / gib
    # Exact SI-table MO blocks retained by DirectCouplingOperator.
    direct_coupling = 8.0 * (
        2 * nc * nc * nv
        + 2 * nc * nv * nv
        + nv * nv
        + nc * nc
        + 3 * nc * nv
        + nc
        + nv
    ) / gib

    ao3 = None
    with_df = method.mrsf.jk.with_df
    if with_df is not None:
        naux = with_df.get_naoaux()
        nao = method.mrsf.ref.nao
        ao3 = 8.0 * naux * nao * (nao + 1) / 2.0 / gib
    backend = getattr(method, "integral_backend", "density-fitting")
    return ResourceEstimate(
        response_dimension=dimension,
        dense_matrix_gib=dense,
        davidson_vectors_gib=davidson,
        ri_coupling_intermediates_gib=(
            ri_coupling if with_df is not None else None
        ),
        direct_coupling_intermediates_gib=(
            direct_coupling if with_df is None else None
        ),
        ao_three_index_gib=ao3,
        integral_backend=backend,
    )
