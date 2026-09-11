"""Density-fitted J/K contractions and EMRSF coupling operations."""

from __future__ import annotations

import numpy as np
from pyscf import df, lib, scf

from .lr import CVSpace
from .space import MRSFSpace


class ResponseJK:
    """Shared direct or RI J/K backend for response-density contractions."""

    def __init__(self, mf, *, density_fit: bool = True, auxbasis=None) -> None:
        self.mf = mf
        self.mol = mf.mol
        self.density_fit = bool(density_fit)
        self.with_df = None
        if self.density_fit:
            existing = getattr(mf, "with_df", None)
            if existing is not None:
                self.with_df = existing
                if auxbasis is not None and existing.auxbasis != auxbasis:
                    self.with_df = df.DF(self.mol, auxbasis=auxbasis)
            else:
                self.with_df = df.DF(self.mol, auxbasis=auxbasis)
            if self.with_df._cderi is None:
                self.with_df.build()

    def get_jk(self, density, *, hermi: int = 0, omega: float | None = None):
        if self.with_df is not None:
            return self.with_df.get_jk(
                np.asarray(density), hermi=hermi, with_j=True, with_k=True, omega=omega
            )
        if omega is None or abs(omega) < 1.0e-15:
            return scf.hf.get_jk(self.mol, density, hermi=hermi)
        with self.mol.with_range_coulomb(omega):
            return scf.hf.get_jk(self.mol, density, hermi=hermi)


def _pair_block(lao, left, right):
    return np.einsum("Puv,ui,vj->Pij", lao, left, right, optimize=True)


def _pair_diagonal(lao, coeff):
    return np.einsum("Puv,ui,vi->Pi", lao, coeff, coeff, optimize=True)


class RICouplingOperator:
    """Matrix-free scaled EMRSF coupling using RI/AO transition densities.

    Only final two-index intermediates are retained.  The potentially large
    three-index C/V tensor is transformed blockwise and discarded during setup.
    ``coupling_scale`` is applied once to the complete Slater--Condon action,
    including every Fock and two-electron contribution.
    """

    backend = "density-fitting"

    def __init__(
        self,
        mrsf_space: MRSFSpace,
        cv_space: CVSpace,
        corrected_fock_mo: np.ndarray,
        *,
        coupling_scale: float,
        jk: ResponseJK,
    ) -> None:
        if mrsf_space.target_multiplicity not in (1, 3):
            raise ValueError("EMRSF coupling supports singlet or triplet targets")
        if jk.with_df is None:
            raise ValueError("RICouplingOperator requires density fitting")
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
        self.jk = jk

        ref = self.ref
        self.cc = ref.coeff[:, ref.closed]
        self.vv = ref.coeff[:, ref.virtual]
        self.co1 = ref.coeff[:, ref.o1 : ref.o1 + 1]
        self.co2 = ref.coeff[:, ref.o2 : ref.o2 + 1]
        nc, nv = len(ref.closed), len(ref.virtual)

        self.a_pp = np.zeros((nc, nv))
        self.b_ps = np.zeros((nc, nv))
        self.d_rq = np.zeros((nc, nv))
        self.e_rq = np.zeros((nc, nv))
        self.oos_singlet = np.zeros((nc, nv))
        self.oos_triplet = np.zeros((nc, nv))
        self.t_s = np.zeros(nv)
        self.u_r = np.zeros(nc)
        self.m_qs = np.zeros((nv, nv))
        self.n_rp = np.zeros((nc, nc))

        nao = ref.nao
        for packed in jk.with_df.loop():
            if packed.shape[1] == nao * (nao + 1) // 2:
                lao = lib.unpack_tril(packed)
            else:
                lao = packed.reshape(-1, nao, nao)

            lcv = _pair_block(lao, self.cc, self.vv)
            lco1 = _pair_block(lao, self.cc, self.co1)[:, :, 0]
            lo2c = _pair_block(lao, self.co2, self.cc)[:, 0, :]
            lvo1 = _pair_block(lao, self.vv, self.co1)[:, :, 0]
            lo2v = _pair_block(lao, self.co2, self.vv)[:, 0, :]
            lpp = _pair_diagonal(lao, self.cc)
            lqq = _pair_diagonal(lao, self.vv)
            lo2o1 = _pair_block(lao, self.co2, self.co1)[:, 0, 0]

            self.a_pp += lpp.T @ lo2v
            self.b_ps += np.einsum("Pps,Pp->ps", lcv, lo2c, optimize=True)
            self.d_rq += np.einsum("Prq,Pq->rq", lcv, lvo1, optimize=True)
            self.e_rq += lco1.T @ lqq
            self.oos_triplet += lco1.T @ lo2v
            self.oos_singlet += lco1.T @ lo2v
            self.oos_singlet -= 2.0 * np.einsum(
                "Prs,P->rs", lcv, lo2o1, optimize=True
            )
            self.t_s += lo2o1 @ lo2v
            self.u_r += lo2o1 @ lco1
            self.m_qs += lvo1.T @ lo2v
            self.n_rp += lco1.T @ lo2c

        self.co1_fock_delta = -self.fock[ref.o2, ref.virtual][None, :]
        self.co1_two_electron_delta = -self.a_pp + self.b_ps
        self.co1_delta = self.co1_fock_delta + self.co1_two_electron_delta
        self.o2v_fock_delta = -self.fock[ref.closed, ref.o1][:, None]
        self.o2v_two_electron_delta = -self.d_rq + self.e_rq
        self.o2v_delta = self.o2v_fock_delta + self.o2v_two_electron_delta
        self.o2v_triplet_fock_delta = -self.o2v_fock_delta
        self.o2v_triplet_two_electron_delta = -self.o2v_two_electron_delta
        self.o2v_triplet_delta = (
            self.o2v_triplet_fock_delta
            + self.o2v_triplet_two_electron_delta
        )
        # Backwards-compatible diagnostic alias for the published singlet row.
        self.oos = self.oos_singlet

    def _cv_matrix(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.cv_space.size:
            raise ValueError(f"Expected a CV vector of length {self.cv_space.size}")
        return vector.reshape((len(self.ref.closed), len(self.ref.virtual)), order="F")

    def _matvec_components_from_jk(self, x, vj, vk):
        ref = self.ref
        jmo = ref.coeff.T @ vj @ ref.coeff
        kmo = ref.coeff.T @ vk @ ref.coeff

        fock = np.zeros((ref.nmo, ref.nmo))
        two_electron = np.zeros((ref.nmo, ref.nmo))
        if self.target_multiplicity == 1:
            fock[ref.o2, ref.o1] = np.sqrt(2.0) * np.sum(
                self.fock[np.ix_(ref.closed, ref.virtual)] * x
            )
            two_electron[ref.o1, ref.o1] = np.sum(self.oos_singlet * x)
            fock[ref.closed, ref.o1] = np.sum(
                self.co1_fock_delta * x, axis=1
            )
            two_electron[ref.closed, ref.o1] = (
                kmo[ref.closed, ref.o2] - 2.0 * jmo[ref.o2, ref.closed]
                + np.sum(self.co1_two_electron_delta * x, axis=1)
            )
            two_electron[ref.closed, ref.o2] = -(x @ self.t_s)
            two_electron[ref.o1, ref.virtual] = self.u_r @ x
            fock[ref.o2, ref.virtual] = np.sum(
                self.o2v_fock_delta * x, axis=0
            )
            two_electron[ref.o2, ref.virtual] = (
                2.0 * jmo[ref.virtual, ref.o1] - kmo[ref.o1, ref.virtual]
                + np.sum(self.o2v_two_electron_delta * x, axis=0)
            )
            two_electron[np.ix_(ref.closed, ref.virtual)] = (
                -x @ self.m_qs.T + self.n_rp.T @ x
            )
        else:
            two_electron[ref.o1, ref.o1] = np.sum(self.oos_triplet * x)
            fock[ref.closed, ref.o1] = np.sum(
                self.co1_fock_delta * x, axis=1
            )
            two_electron[ref.closed, ref.o1] = (
                kmo[ref.closed, ref.o2]
                + np.sum(self.co1_two_electron_delta * x, axis=1)
            )
            two_electron[ref.closed, ref.o2] = -(x @ self.t_s)
            two_electron[ref.o1, ref.virtual] = -(self.u_r @ x)
            fock[ref.o2, ref.virtual] = np.sum(
                self.o2v_triplet_fock_delta * x, axis=0
            )
            two_electron[ref.o2, ref.virtual] = (
                kmo[ref.o1, ref.virtual]
                + np.sum(self.o2v_triplet_two_electron_delta * x, axis=0)
            )
            two_electron[np.ix_(ref.closed, ref.virtual)] = (
                -x @ self.m_qs.T - self.n_rp.T @ x
            )
        scale = self.coupling_scale
        return {
            "fock": scale * self.mrsf_space.pack(fock),
            "two_electron": scale * self.mrsf_space.pack(two_electron),
        }

    def _matvec_from_jk(self, x, vj, vk):
        components = self._matvec_components_from_jk(x, vj, vk)
        return components["fock"] + components["two_electron"]

    def _rmatvec_components_from_jk(self, y, co1_jk=None, o2v_jk=None):
        ref = self.ref
        fock = np.zeros((len(ref.closed), len(ref.virtual)))
        two_electron = np.zeros_like(fock)

        if self.target_multiplicity == 1:
            fock += np.sqrt(2.0) * y[ref.o2, ref.o1] * self.fock[
                np.ix_(ref.closed, ref.virtual)
            ]
            two_electron += y[ref.o1, ref.o1] * self.oos_singlet

            co1 = y[ref.closed, ref.o1]
            fock += co1[:, None] * self.co1_fock_delta
            if co1_jk is not None:
                vj, vk = co1_jk
                two_electron += (
                    self.cc.T @ vk @ self.vv - 2.0 * self.cc.T @ vj @ self.vv
                )
                two_electron += co1[:, None] * self.co1_two_electron_delta

            two_electron += -y[ref.closed, ref.o2][:, None] * self.t_s[None, :]
            two_electron += self.u_r[:, None] * y[ref.o1, ref.virtual][None, :]

            o2v = y[ref.o2, ref.virtual]
            fock += self.o2v_fock_delta * o2v[None, :]
            if o2v_jk is not None:
                vj, vk = o2v_jk
                two_electron += (
                    2.0 * self.cc.T @ vj @ self.vv - self.cc.T @ vk @ self.vv
                )
                two_electron += self.o2v_two_electron_delta * o2v[None, :]

            cv = y[np.ix_(ref.closed, ref.virtual)]
            two_electron += -cv @ self.m_qs + self.n_rp @ cv
        else:
            two_electron += y[ref.o1, ref.o1] * self.oos_triplet
            co1 = y[ref.closed, ref.o1]
            fock += co1[:, None] * self.co1_fock_delta
            if co1_jk is not None:
                _, vk = co1_jk
                two_electron += self.cc.T @ vk @ self.vv
                two_electron += co1[:, None] * self.co1_two_electron_delta
            two_electron += -y[ref.closed, ref.o2][:, None] * self.t_s[None, :]
            two_electron -= self.u_r[:, None] * y[ref.o1, ref.virtual][None, :]
            o2v = y[ref.o2, ref.virtual]
            fock += self.o2v_triplet_fock_delta * o2v[None, :]
            if o2v_jk is not None:
                _, vk = o2v_jk
                two_electron += self.cc.T @ vk @ self.vv
                two_electron += (
                    self.o2v_triplet_two_electron_delta * o2v[None, :]
                )
            cv = y[np.ix_(ref.closed, ref.virtual)]
            two_electron += -cv @ self.m_qs - self.n_rp @ cv
        scale = self.coupling_scale
        return {
            "fock": scale * fock.reshape(-1, order="F"),
            "two_electron": scale * two_electron.reshape(-1, order="F"),
        }

    def _rmatvec_from_jk(self, y, co1_jk=None, o2v_jk=None):
        components = self._rmatvec_components_from_jk(
            y, co1_jk=co1_jk, o2v_jk=o2v_jk
        )
        return components["fock"] + components["two_electron"]

    def matvec_components(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Return Fock and two-electron contributions to ``C @ vector``."""

        if self.cv_space.size == 0:
            zero = np.zeros(self.mrsf_space.size)
            return {"fock": zero.copy(), "two_electron": zero.copy()}
        x = self._cv_matrix(vector)
        dm = self.cc @ x @ self.vv.T
        vj, vk = self.jk.get_jk(dm, hermi=0)
        return self._matvec_components_from_jk(x, vj, vk)

    def rmatvec_components(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Return Fock and two-electron contributions to ``C.T @ vector``."""

        if self.cv_space.size == 0:
            zero = np.zeros(0)
            return {"fock": zero.copy(), "two_electron": zero.copy()}
        y = self.mrsf_space.unpack(self.mrsf_space.mask_redundant(vector))
        densities = []
        tags = []
        co1 = y[self.ref.closed, self.ref.o1]
        if np.any(co1):
            densities.append((self.cc @ co1)[:, None] @ self.co2.T)
            tags.append("co1")
        o2v = y[self.ref.o2, self.ref.virtual]
        if np.any(o2v):
            densities.append(self.co1 @ (self.vv @ o2v)[None, :])
            tags.append("o2v")
        blocks = {}
        if densities:
            vj, vk = self.jk.get_jk(np.asarray(densities), hermi=0)
            blocks = {tag: (vj[k], vk[k]) for k, tag in enumerate(tags)}
        return self._rmatvec_components_from_jk(
            y, co1_jk=blocks.get("co1"), o2v_jk=blocks.get("o2v")
        )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply ``C`` to an added-CV vector, returning the full MRSF vector."""

        if self.cv_space.size == 0:
            return np.zeros(self.mrsf_space.size)
        x = self._cv_matrix(vector)
        dm = self.cc @ x @ self.vv.T
        vj, vk = self.jk.get_jk(dm, hermi=0)
        return self._matvec_from_jk(x, vj, vk)

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply the exact algebraic adjoint ``C.T`` without forming ``C``."""

        if self.cv_space.size == 0:
            return np.zeros(0)
        y = self.mrsf_space.unpack(self.mrsf_space.mask_redundant(vector))
        densities = []
        tags = []
        co1 = y[self.ref.closed, self.ref.o1]
        if np.any(co1):
            densities.append((self.cc @ co1)[:, None] @ self.co2.T)
            tags.append("co1")
        o2v = y[self.ref.o2, self.ref.virtual]
        if np.any(o2v):
            densities.append(self.co1 @ (self.vv @ o2v)[None, :])
            tags.append("o2v")
        blocks = {}
        if densities:
            vj, vk = self.jk.get_jk(np.asarray(densities), hermi=0)
            blocks = {tag: (vj[k], vk[k]) for k, tag in enumerate(tags)}
        return self._rmatvec_from_jk(
            y, co1_jk=blocks.get("co1"), o2v_jk=blocks.get("o2v")
        )

    def apply(self, cv_vector: np.ndarray, mrsf_vector: np.ndarray):
        """Apply ``C`` and ``C.T`` together with one batched RI-JK dispatch."""

        if self.cv_space.size == 0:
            return np.zeros(self.mrsf_space.size), np.zeros(0)
        x = self._cv_matrix(cv_vector)
        y = self.mrsf_space.unpack(self.mrsf_space.mask_redundant(mrsf_vector))
        densities = [self.cc @ x @ self.vv.T]
        tags = ["cv"]
        co1 = y[self.ref.closed, self.ref.o1]
        if np.any(co1):
            densities.append((self.cc @ co1)[:, None] @ self.co2.T)
            tags.append("co1")
        o2v = y[self.ref.o2, self.ref.virtual]
        if np.any(o2v):
            densities.append(self.co1 @ (self.vv @ o2v)[None, :])
            tags.append("o2v")

        vj, vk = self.jk.get_jk(np.asarray(densities), hermi=0)
        blocks = {tag: (vj[k], vk[k]) for k, tag in enumerate(tags)}
        c_x = self._matvec_from_jk(x, *blocks["cv"])
        ct_y = self._rmatvec_from_jk(
            y, co1_jk=blocks.get("co1"), o2v_jk=blocks.get("o2v")
        )
        return c_x, ct_y
