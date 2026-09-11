"""MRSF--LR(CV) coupling from Oh et al., SI Tables S1--S3."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyscf import ao2mo

from .lr import CVSpace
from .space import MRSFSpace


class _MOIntegrals:
    """Small-block AO-to-MO integral transformer.

    The validation implementation avoids an all-MO four-index tensor.  Each
    block is generated once and cached.  A density-fitted/matrix-free version
    should replace this class for production-size chromophores.
    """

    def __init__(self, ref) -> None:
        self.ref = ref

    @lru_cache(maxsize=None)
    def block(self, i1: tuple[int, ...], i2: tuple[int, ...], i3: tuple[int, ...], i4: tuple[int, ...]):
        shape = (len(i1), len(i2), len(i3), len(i4))
        if 0 in shape:
            return np.zeros(shape)
        c = self.ref.coeff
        value = ao2mo.general(
            self.ref.mf.mol,
            (c[:, i1], c[:, i2], c[:, i3], c[:, i4]),
            compact=False,
        )
        return np.asarray(value).reshape(shape)


class DirectCouplingOperator:
    """Exact four-centre, matrix-free MRSF--CV coupling operator.

    This is the non-RI counterpart of :class:`RICouplingOperator`.  The ten
    MO-integral blocks that occur in SI Tables S1--S3 are transformed exactly
    once with PySCF's four-centre AO-to-MO machinery.  Davidson applications
    then use tensor contractions and never assemble the full rectangular
    coupling matrix.

    The implementation deliberately exposes the same ``matvec``, ``rmatvec``
    and component-decomposition interface as the RI backend.  Consequently
    selecting direct integrals changes only the numerical integral backend;
    it does not change the MRSF equations, spin adaptation, S30 correction, or
    the published ``c_cp`` scaling.
    """

    backend = "direct-four-centre"

    def __init__(
        self,
        mrsf_space: MRSFSpace,
        cv_space: CVSpace,
        corrected_fock_mo: np.ndarray,
        *,
        coupling_scale: float,
    ) -> None:
        if mrsf_space.target_multiplicity not in (1, 3):
            raise ValueError("EMRSF coupling supports singlet or triplet targets")
        if mrsf_space.ref is not cv_space.ref:
            raise ValueError("MRSF and CV spaces must share the same reference")

        self.mrsf_space = mrsf_space
        self.target_multiplicity = int(mrsf_space.target_multiplicity)
        self.cv_space = cv_space
        self.ref = mrsf_space.ref
        self.fock = np.asarray(corrected_fock_mo)
        self.coupling_scale = float(coupling_scale)
        if not np.isfinite(self.coupling_scale):
            raise ValueError("coupling_scale must be finite")

        ref = self.ref
        closed = tuple(map(int, ref.closed))
        virtual = tuple(map(int, ref.virtual))
        o1, o2 = ref.o1, ref.o2
        integrals = _MOIntegrals(ref)

        # The comments reproduce the orbital order used in SI Tables S1--S3.
        self.cv_oc = integrals.block(closed, virtual, (o2,), closed)[:, :, 0, :]
        self.cc_vo = integrals.block(closed, closed, virtual, (o2,))[:, :, :, 0]
        self.oo_ov = integrals.block((o2,), (o1,), (o2,), virtual)[0, 0, 0, :]
        self.cv_vo = integrals.block(closed, virtual, virtual, (o1,))[:, :, :, 0]
        self.co_vv = integrals.block(closed, (o1,), virtual, virtual)[:, 0, :, :]
        co_ov = integrals.block(closed, (o1,), (o2,), virtual)[:, 0, 0, :]
        self.vo_ov = integrals.block(virtual, (o1,), (o2,), virtual)[:, 0, 0, :]
        self.co_oc = integrals.block(closed, (o1,), (o2,), closed)[:, 0, 0, :]
        self.co_oo = integrals.block(closed, (o1,), (o2,), (o1,))[:, 0, 0, 0]
        cv_oo = integrals.block(closed, virtual, (o2,), (o1,))[:, :, 0, 0]
        self.oos_singlet = co_ov - 2.0 * cv_oo
        self.oos_triplet = co_ov
        self.oos = self.oos_singlet

        nc = len(closed)
        nv = len(virtual)
        if nc:
            self.co1_diagonal_delta = np.stack(
                [
                    -self.cc_vo[p, p, :] + self.cv_oc[p, :, p]
                    for p in range(nc)
                ]
            )
        else:
            self.co1_diagonal_delta = np.zeros((0, nv))
        self.o2v_diagonal_delta = (
            -np.diagonal(self.cv_vo, axis1=1, axis2=2)
            + np.diagonal(self.co_vv, axis1=1, axis2=2)
        )
        self.o2v_triplet_diagonal_delta = -self.o2v_diagonal_delta

    def _cv_matrix(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.cv_space.size:
            raise ValueError(f"Expected a CV vector of length {self.cv_space.size}")
        return vector.reshape(
            (len(self.ref.closed), len(self.ref.virtual)), order="F"
        )

    def _mrsf_matrix(self, vector: np.ndarray) -> np.ndarray:
        vector = self.mrsf_space.mask_redundant(np.asarray(vector, dtype=float))
        return self.mrsf_space.unpack(vector)

    def _forward_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        ref = self.ref
        fock = np.zeros((ref.nmo, ref.nmo))
        two = np.zeros_like(fock)

        if self.target_multiplicity == 1:
            fock[ref.o2, ref.o1] = np.sqrt(2.0) * np.sum(
                self.fock[np.ix_(ref.closed, ref.virtual)] * x
            )
            two[ref.o1, ref.o1] = np.sum(self.oos_singlet * x)

            fock[ref.closed, ref.o1] = -x @ self.fock[ref.o2, ref.virtual]
            two[ref.closed, ref.o1] = (
                np.einsum("rps,rs->p", self.cc_vo, x, optimize=True)
                - 2.0 * np.einsum("rsp,rs->p", self.cv_oc, x, optimize=True)
                + np.sum(self.co1_diagonal_delta * x, axis=1)
            )
            two[ref.closed, ref.o2] = -(x @ self.oo_ov)
            two[ref.o1, ref.virtual] = self.co_oo @ x

            fock[ref.o2, ref.virtual] = -(
                self.fock[ref.closed, ref.o1] @ x
            )
            two[ref.o2, ref.virtual] = (
                2.0 * np.einsum("rsq,rs->q", self.cv_vo, x, optimize=True)
                - np.einsum("rqs,rs->q", self.co_vv, x, optimize=True)
                + np.sum(self.o2v_diagonal_delta * x, axis=0)
            )
            two[np.ix_(ref.closed, ref.virtual)] = (
                -x @ self.vo_ov.T + self.co_oc.T @ x
            )
        else:
            # Triplet spin adaptation combines the two determinant rows as
            # S1+S2.  G and D are absent from the live triplet MRSF space.
            two[ref.o1, ref.o1] = np.sum(self.oos_triplet * x)
            fock[ref.closed, ref.o1] = -x @ self.fock[ref.o2, ref.virtual]
            two[ref.closed, ref.o1] = (
                np.einsum("rps,rs->p", self.cc_vo, x, optimize=True)
                + np.sum(self.co1_diagonal_delta * x, axis=1)
            )
            two[ref.closed, ref.o2] = -(x @ self.oo_ov)
            two[ref.o1, ref.virtual] = -(self.co_oo @ x)
            fock[ref.o2, ref.virtual] = self.fock[ref.closed, ref.o1] @ x
            two[ref.o2, ref.virtual] = (
                np.einsum("rqs,rs->q", self.co_vv, x, optimize=True)
                + np.sum(self.o2v_triplet_diagonal_delta * x, axis=0)
            )
            two[np.ix_(ref.closed, ref.virtual)] = (
                -x @ self.vo_ov.T - self.co_oc.T @ x
            )

        scale = self.coupling_scale
        return {
            "fock": scale * self.mrsf_space.pack(fock),
            "two_electron": scale * self.mrsf_space.pack(two),
        }

    def _reverse_components(self, y: np.ndarray) -> dict[str, np.ndarray]:
        ref = self.ref
        fock = np.zeros((len(ref.closed), len(ref.virtual)))
        two = np.zeros_like(fock)

        if self.target_multiplicity == 1:
            fock += np.sqrt(2.0) * y[ref.o2, ref.o1] * self.fock[
                np.ix_(ref.closed, ref.virtual)
            ]
            two += y[ref.o1, ref.o1] * self.oos_singlet

            y_co1 = y[ref.closed, ref.o1]
            fock += -y_co1[:, None] * self.fock[ref.o2, ref.virtual][None, :]
            two += np.einsum("p,rps->rs", y_co1, self.cc_vo, optimize=True)
            two -= 2.0 * np.einsum("p,rsp->rs", y_co1, self.cv_oc, optimize=True)
            two += y_co1[:, None] * self.co1_diagonal_delta

            two += -y[ref.closed, ref.o2][:, None] * self.oo_ov[None, :]
            two += self.co_oo[:, None] * y[ref.o1, ref.virtual][None, :]

            y_o2v = y[ref.o2, ref.virtual]
            fock += -self.fock[ref.closed, ref.o1][:, None] * y_o2v[None, :]
            two += 2.0 * np.einsum("q,rsq->rs", y_o2v, self.cv_vo, optimize=True)
            two -= np.einsum("q,rqs->rs", y_o2v, self.co_vv, optimize=True)
            two += self.o2v_diagonal_delta * y_o2v[None, :]

            y_cv = y[np.ix_(ref.closed, ref.virtual)]
            two += -y_cv @ self.vo_ov + self.co_oc @ y_cv
        else:
            two += y[ref.o1, ref.o1] * self.oos_triplet
            y_co1 = y[ref.closed, ref.o1]
            fock += -y_co1[:, None] * self.fock[ref.o2, ref.virtual][None, :]
            two += np.einsum("p,rps->rs", y_co1, self.cc_vo, optimize=True)
            two += y_co1[:, None] * self.co1_diagonal_delta
            two += -y[ref.closed, ref.o2][:, None] * self.oo_ov[None, :]
            two -= self.co_oo[:, None] * y[ref.o1, ref.virtual][None, :]
            y_o2v = y[ref.o2, ref.virtual]
            fock += self.fock[ref.closed, ref.o1][:, None] * y_o2v[None, :]
            two += np.einsum("q,rqs->rs", y_o2v, self.co_vv, optimize=True)
            two += self.o2v_triplet_diagonal_delta * y_o2v[None, :]
            y_cv = y[np.ix_(ref.closed, ref.virtual)]
            two += -y_cv @ self.vo_ov - self.co_oc @ y_cv

        scale = self.coupling_scale
        return {
            "fock": scale * fock.reshape(-1, order="F"),
            "two_electron": scale * two.reshape(-1, order="F"),
        }

    def matvec_components(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        if self.cv_space.size == 0:
            zero = np.zeros(self.mrsf_space.size)
            return {"fock": zero.copy(), "two_electron": zero.copy()}
        return self._forward_components(self._cv_matrix(vector))

    def rmatvec_components(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        if self.cv_space.size == 0:
            zero = np.zeros(0)
            return {"fock": zero.copy(), "two_electron": zero.copy()}
        return self._reverse_components(self._mrsf_matrix(vector))

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        components = self.matvec_components(vector)
        return components["fock"] + components["two_electron"]

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        components = self.rmatvec_components(vector)
        return components["fock"] + components["two_electron"]

    def apply(self, cv_vector: np.ndarray, mrsf_vector: np.ndarray):
        return self.matvec(cv_vector), self.rmatvec(mrsf_vector)


def build_coupling(
    mrsf_space: MRSFSpace,
    cv_space: CVSpace,
    corrected_fock_mo: np.ndarray,
    *,
    coupling_scale: float,
) -> np.ndarray:
    """Return the scaled spin-adapted MRSF--CV coupling ``c_cp * C``.

    The raw :math:`M_s=+1/-1` elements in SI Tables S1 and S2 are combined
    with S1-S2 for a singlet target or S1+S2 for a triplet target.  The special
    open-shell configurations use the corresponding spin-adapted rows.  In the
    EMRSF block equation the coupling
    coefficient :math:`c_{cp}` multiplies the *whole* Slater--Condon matrix,
    including both Fock and two-electron terms.  Oh et al. set
    :math:`c_{cp}=c_H`, the global Hartree--Fock exchange fraction.

    The :math:`\\sqrt{2}` factor of the singlet Table-S3 G row and both
    spin-adapted combinations are fixed parts of the method, not adjustable
    phase conventions.  The triplet extension has no G or D row.
    """

    if mrsf_space.target_multiplicity not in (1, 3):
        raise ValueError("EMRSF coupling supports singlet or triplet targets")
    ref = mrsf_space.ref
    if cv_space.ref is not ref:
        raise ValueError("MRSF and CV spaces must share the same reference")

    closed = tuple(map(int, ref.closed))
    virtual = tuple(map(int, ref.virtual))
    o1, o2 = ref.o1, ref.o2
    ints = _MOIntegrals(ref)

    # Blocks are named by their orbital domains.  The array indices follow the
    # symbols p,q (MRSF) and r,s (LR) used in Tables S1--S3.
    cv_oc = ints.block(closed, virtual, (o2,), closed)[:, :, 0, :]       # r,s,O2,p
    cc_vo = ints.block(closed, closed, virtual, (o2,))[:, :, :, 0]      # r,p,s,O2
    oo_ov = ints.block((o2,), (o1,), (o2,), virtual)[0, 0, 0, :]        # O2,O1,O2,s
    cv_vo = ints.block(closed, virtual, virtual, (o1,))[:, :, :, 0]     # r,s,q,O1
    co_vv = ints.block(closed, (o1,), virtual, virtual)[:, 0, :, :]     # r,O1,q,s
    co_ov = ints.block(closed, (o1,), (o2,), virtual)[:, 0, 0, :]       # r,O1,O2,s
    vo_ov = ints.block(virtual, (o1,), (o2,), virtual)[:, 0, 0, :]      # q,O1,O2,s
    co_oc = ints.block(closed, (o1,), (o2,), closed)[:, 0, 0, :]        # r,O1,O2,p
    co_oo = ints.block(closed, (o1,), (o2,), (o1,))[:, 0, 0, 0]         # r,O1,O2,O1
    cv_oo = ints.block(closed, virtual, (o2,), (o1,))[:, :, 0, 0]       # r,s,O2,O1

    cpos = {p: k for k, p in enumerate(closed)}
    vpos = {q: k for k, q in enumerate(virtual)}
    coupling = np.zeros((mrsf_space.size, cv_space.size))
    scale = float(coupling_scale)
    if not np.isfinite(scale):
        raise ValueError("coupling_scale must be finite")

    for im, (i, a) in enumerate(mrsf_space.labels):
        for jl, (r, s) in enumerate(cv_space.labels):
            ir, js = cpos[r], vpos[s]
            value = 0.0

            if mrsf_space.target_multiplicity == 1:
                if (i, a) == (o2, o1):                     # G
                    value = np.sqrt(2.0) * corrected_fock_mo[r, s]
                elif (i, a) == (o1, o2):                   # D
                    value = 0.0
                elif (i, a) == (o1, o1):                   # (L-R)/sqrt(2), OOS
                    value = co_ov[ir, js] - 2.0 * cv_oo[ir, js]
                elif i in cpos and a == o1:                # CO1, S1-S2
                    ip = cpos[i]
                    if i == r:
                        value = -corrected_fock_mo[o2, s] - cv_oc[ir, js, ip]
                    else:
                        value = cc_vo[ir, ip, js] - 2.0 * cv_oc[ir, js, ip]
                elif i in cpos and a == o2:                # CO2
                    if i == r:
                        value = -oo_ov[js]
                elif i == o1 and a in vpos:                # O1V
                    if a == s:
                        value = co_oo[ir]
                elif i == o2 and a in vpos:                # O2V
                    iq = vpos[a]
                    direct = cv_vo[ir, js, iq]
                    if a == s:
                        value = direct - corrected_fock_mo[r, o1]
                    else:
                        value = 2.0 * direct - co_vv[ir, iq, js]
                elif i in cpos and a in vpos:              # MRSF-CV
                    ip, iq = cpos[i], vpos[a]
                    if i == r:
                        value -= vo_ov[iq, js]
                    if a == s:
                        value += co_oc[ir, ip]
            else:
                if (i, a) == (o1, o1):                    # (L+R)/sqrt(2), OOS
                    value = co_ov[ir, js]
                elif i in cpos and a == o1:                # CO1, S1+S2
                    ip = cpos[i]
                    if i == r:
                        value = -corrected_fock_mo[o2, s] + cv_oc[ir, js, ip]
                    else:
                        value = cc_vo[ir, ip, js]
                elif i in cpos and a == o2:                # CO2
                    if i == r:
                        value = -oo_ov[js]
                elif i == o1 and a in vpos:                # O1V
                    if a == s:
                        value = -co_oo[ir]
                elif i == o2 and a in vpos:                # O2V
                    iq = vpos[a]
                    if a == s:
                        value = corrected_fock_mo[r, o1] + cv_vo[ir, js, iq]
                    else:
                        value = co_vv[ir, iq, js]
                elif i in cpos and a in vpos:              # MRSF-CV
                    ip, iq = cpos[i], vpos[a]
                    if i == r:
                        value -= vo_ov[iq, js]
                    if a == s:
                        value -= co_oc[ir, ip]

            coupling[im, jl] = scale * value
    return coupling
