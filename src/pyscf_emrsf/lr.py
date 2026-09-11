"""Closed-shell LR-TDA sector used by EMRSF-TDDFT.

The orbitals are those of the high-spin triplet reference.  They are occupied
as C/O1 (doubly occupied) and O2/V (empty) only while evaluating the LR kernel.
This is not a second SCF calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import dft, scf, tdscf  # noqa: F401 - installs gen_response methods

from .kernels import exchange_model
from .reference import TripletReference


@dataclass(frozen=True)
class CVSpace:
    """The additional closed-to-virtual configurations of EMRSF."""

    ref: TripletReference

    @property
    def labels(self) -> tuple[tuple[int, int], ...]:
        # Column-major convention, as in MRSFSpace.
        return tuple((int(i), int(a)) for a in self.ref.virtual for i in self.ref.closed)

    @property
    def size(self) -> int:
        return len(self.ref.closed) * len(self.ref.virtual)

    def unpack(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.size:
            raise ValueError(f"Expected a vector of length {self.size}")
        out = np.zeros((self.ref.nmo, self.ref.nmo))
        for value, (i, a) in zip(vector, self.labels):
            out[i, a] = value
        return out

    def pack(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray([matrix[i, a] for i, a in self.labels])


def _closed_shell_driver(ref: TripletReference, *, response_jk=None):
    """Create a non-iterated RHF/RKS object at the C/O1 density."""

    mol = ref.mf.mol.copy()
    mol.spin = 0
    half = mol.nelectron // 2
    mol.nelec = (half, half)

    if hasattr(ref.mf, "xc"):
        driver = dft.RKS(mol)
        driver.xc = ref.mf.xc
        driver._numint = ref.mf._numint
        # Reuse the converged quadrature.  The atom grid is independent of spin.
        driver.grids = ref.mf.grids
        if hasattr(ref.mf, "nlc"):
            driver.nlc = ref.mf.nlc
    else:
        driver = scf.RHF(mol)

    if response_jk is not None and response_jk.with_df is not None:
        driver = driver.density_fit(auxbasis=response_jk.with_df.auxbasis)
        driver.with_df = response_jk.with_df

    occupation = np.zeros(ref.nmo)
    occupation[ref.closed] = 2.0
    occupation[ref.o1] = 2.0
    driver.mo_coeff = ref.coeff
    driver.mo_occ = occupation
    driver.mo_energy = np.asarray(ref.energy).copy()
    return driver


class CorrectedLRTDA:
    """Spin-adapted LR-CV TDA action in the triplet-reference orbital basis.

    The diagonal Fock correction is eq. S30 of Oh *et al.*.  Setting
    ``apply_correction=False`` reproduces the deliberately uncorrected model
    used for the LiF diagnostic in their Supporting Information.
    """

    def __init__(
        self,
        ref: TripletReference,
        *,
        target_multiplicity: int = 1,
        apply_correction: bool = True,
        response_jk=None,
    ) -> None:
        self.ref = ref
        self.space = CVSpace(ref)
        self.driver = _closed_shell_driver(ref, response_jk=response_jk)
        self.target_multiplicity = int(target_multiplicity)
        if self.target_multiplicity not in (1, 3):
            raise ValueError("target_multiplicity must be 1 or 3")
        self.apply_correction = bool(apply_correction)

        model = exchange_model(ref.mf)
        if abs(model.omega) > 1.0e-14:
            raise NotImplementedError(
                "Oh et al. derive eq. S30 for a global hybrid coefficient c_H; "
                "range-separated EMRSF is not enabled in this validation release"
            )
        self.exact_exchange = model.short_range

        dm0 = self.driver.make_rdm1()
        fock_ao = self.driver.get_fock(dm=dm0)
        solvent = getattr(ref.mf, "with_solvent", None)
        if solvent is not None and getattr(solvent, "v", None) is not None:
            # The published MRSF/EMRSF response has no PCM kernel.  Retain the
            # converged triplet reaction potential as a frozen one-electron
            # field in the artificial closed-shell LR block.
            fock_ao = fock_ao + np.asarray(solvent.v)
        self.fock_mo_raw = ref.coeff.T @ fock_ao @ ref.coeff
        self.s30_correction = np.zeros(ref.nmo)
        self.fock_mo = self.fock_mo_raw.copy()
        self.response_jk = response_jk
        if self.apply_correction:
            d_open = np.outer(ref.coeff[:, ref.o2], ref.coeff[:, ref.o2])
            d_open -= np.outer(ref.coeff[:, ref.o1], ref.coeff[:, ref.o1])
            coulomb = self.driver.get_j(self.driver.mol, d_open, hermi=1)
            self.s30_correction = (1.0 - self.exact_exchange) * np.diag(
                ref.coeff.T @ coulomb @ ref.coeff
            )
            self.fock_mo[np.diag_indices(ref.nmo)] += self.s30_correction

        self._response = (
            self.driver.gen_response(singlet=True, hermi=0)
            if self.target_multiplicity == 1
            else None
        )

    def _exchange_response(self, density: np.ndarray) -> np.ndarray:
        """Return ``-0.5*c_H*K[density]`` for the triplet LR block."""

        if self.response_jk is None:
            _, exchange = scf.hf.get_jk(self.driver.mol, density, hermi=0)
        else:
            _, exchange = self.response_jk.get_jk(density, hermi=0)
        return -0.5 * self.exact_exchange * np.asarray(exchange)

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        x = self.space.unpack(vector)
        ref = self.ref
        closed, virtual = ref.closed, ref.virtual
        block = x[np.ix_(closed, virtual)]

        out = np.zeros_like(x)
        out[np.ix_(closed, virtual)] = (
            block @ self.fock_mo[np.ix_(virtual, virtual)].T
            - self.fock_mo[np.ix_(closed, closed)] @ block
        )

        # The factor two is paired with PySCF's -1/2 exchange convention.
        dm1 = 2.0 * ref.coeff[:, virtual] @ block.T @ ref.coeff[:, closed].T
        if self.target_multiplicity == 1:
            v1 = self._response(dm1)
        else:
            v1 = self._exchange_response(dm1)
        # The response density is non-Hermitian.  This orientation is essential:
        # it is the transpose of C_o^T V C_v used for an ordinary AO density.
        response = (ref.coeff[:, virtual].T @ v1 @ ref.coeff[:, closed]).T
        out[np.ix_(closed, virtual)] += response
        return self.space.pack(out)

    def diagonal_guess(self) -> np.ndarray:
        """Fock-difference preconditioner for the added C->V sector."""

        return np.asarray(
            [self.fock_mo[a, a] - self.fock_mo[i, i] for i, a in self.space.labels]
        )

    def diagonal_components(self, *, batch_size: int = 32) -> dict[str, np.ndarray]:
        """Decompose every LR-CV diagonal element at equation level.

        The returned arrays follow ``CVSpace.labels`` and separate the raw
        closed-shell Fock difference, the diagonal correction of SI eq. S30,
        the Coulomb response, global exact exchange, and the residual
        semilocal XC kernel.  Their sum is the diagonal of ``A_CV`` before the
        common EMRSF offset ``A_G`` is added.

        ``batch_size`` limits the temporary AO response densities.  This
        routine is intended for validation logs; it is not used by Davidson.
        """

        size = self.space.size
        if size == 0:
            empty = np.empty(0)
            return {
                "raw_fock": empty.copy(),
                "s30": empty.copy(),
                "coulomb": empty.copy(),
                "exact_exchange": empty.copy(),
                "xc_kernel": empty.copy(),
                "lr_diagonal": empty.copy(),
            }
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        labels = self.space.labels
        raw_fock = np.asarray(
            [
                self.fock_mo_raw[a, a] - self.fock_mo_raw[i, i]
                for i, a in labels
            ],
            dtype=float,
        )
        s30 = np.asarray(
            [self.s30_correction[a] - self.s30_correction[i] for i, a in labels],
            dtype=float,
        )
        coulomb = np.empty(size)
        exact_exchange = np.empty(size)
        xc_kernel = np.empty(size)
        coeff = np.asarray(self.ref.coeff)

        def as_batch(value, expected: int, name: str) -> np.ndarray:
            matrices = np.asarray(value)
            if matrices.ndim == 2:
                matrices = matrices[None, :, :]
            if matrices.ndim != 3 or matrices.shape[0] != expected:
                raise RuntimeError(
                    f"Unexpected {name} response shape {matrices.shape}; "
                    f"expected ({expected}, nao, nao)"
                )
            return matrices

        for first in range(0, size, batch_size):
            last = min(size, first + batch_size)
            batch_labels = labels[first:last]
            occupied = np.asarray([i for i, _ in batch_labels], dtype=int)
            virtual = np.asarray([a for _, a in batch_labels], dtype=int)
            ci = coeff[:, occupied]
            ca = coeff[:, virtual]

            # This is exactly the non-Hermitian singlet response density used
            # by matvec: 2 C_a X_ai C_i^T for one unit configuration.
            densities = 2.0 * np.einsum(
                "pb,qb->bpq", ca, ci.conj(), optimize=True
            )
            response = None
            if self.target_multiplicity == 1:
                response = as_batch(
                    self._response(densities), last - first, "total LR"
                )
            if self.response_jk is None:
                vj, vk = scf.hf.get_jk(
                    self.driver.mol, densities, hermi=0
                )
            else:
                vj, vk = self.response_jk.get_jk(densities, hermi=0)
            vj = as_batch(vj, last - first, "Coulomb")
            vk = as_batch(vk, last - first, "exchange")

            def project(matrices: np.ndarray) -> np.ndarray:
                values = np.einsum(
                    "pb,bpq,qb->b", ca.conj(), matrices, ci, optimize=True
                )
                return np.asarray(np.real_if_close(values), dtype=float)

            if response is None:
                total_response = None
                coulomb[first:last] = 0.0
            else:
                total_response = project(response)
                coulomb[first:last] = project(vj)
            exact_exchange[first:last] = (
                -0.5 * self.exact_exchange * project(vk)
            )
            if total_response is None:
                xc_kernel[first:last] = 0.0
            else:
                xc_kernel[first:last] = (
                    total_response
                    - coulomb[first:last]
                    - exact_exchange[first:last]
                )

        lr_diagonal = raw_fock + s30 + coulomb + exact_exchange + xc_kernel
        return {
            "raw_fock": raw_fock,
            "s30": s30,
            "coulomb": coulomb,
            "exact_exchange": exact_exchange,
            "xc_kernel": xc_kernel,
            "lr_diagonal": lr_diagonal,
        }

    def build_matrix(self, *, symmetrise: bool = True) -> np.ndarray:
        if self.space.size == 0:
            return np.zeros((0, 0))
        eye = np.eye(self.space.size)
        matrix = np.column_stack([self.matvec(eye[:, k]) for k in range(self.space.size)])
        if symmetrise:
            matrix = 0.5 * (matrix + matrix.T)
        return matrix
