"""Spin-adapted MRSF response-space packing and AO/MO transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import TripletReference


@dataclass(frozen=True)
class MRSFSpace:
    """Packed MRSF singlet or triplet response space."""

    ref: TripletReference
    target_multiplicity: int = 1

    def __post_init__(self) -> None:
        if self.target_multiplicity not in (1, 3):
            raise ValueError("target_multiplicity must be 1 or 3")

    @property
    def rows(self) -> np.ndarray:
        nalpha = len(self.ref.closed) + 2
        return np.arange(nalpha, dtype=int)

    @property
    def cols(self) -> np.ndarray:
        # beta-unoccupied orbitals are O1, O2, and V
        return np.arange(len(self.ref.closed), self.ref.nmo, dtype=int)

    @property
    def raw_shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.cols)

    @property
    def size(self) -> int:
        return self.raw_shape[0] * self.raw_shape[1]

    @property
    def labels(self) -> tuple[tuple[int, int], ...]:
        # Matches the column-major MRSF ordering used in the published/OpenQP
        # formulation: beta target outer loop, alpha source inner loop.
        return tuple((i, a) for a in self.cols for i in self.rows)

    def unpack(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.size:
            raise ValueError(f"Expected a vector of length {self.size}")
        matrix = np.zeros((self.ref.nmo, self.ref.nmo))
        for value, (i, a) in zip(vector, self.labels):
            matrix[i, a] = value
        return matrix

    def pack(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        return np.asarray([matrix[i, a] for i, a in self.labels])

    def mask_redundant(self, vector: np.ndarray) -> np.ndarray:
        """Project out response slots absent from the chosen spin symmetry."""

        out = np.array(vector, dtype=float, copy=True)
        x = self.unpack(out)
        o1, o2 = self.ref.o1, self.ref.o2
        x[o2, o2] = 0.0
        if self.target_multiplicity == 3:
            x[o1, o2] = 0.0
            x[o2, o1] = 0.0
        return self.pack(x)

    def diagonal_guess(self) -> np.ndarray:
        ea = self.ref.energy
        diagonal = np.empty(self.size)
        o1, o2 = self.ref.o1, self.ref.o2
        for k, (i, a) in enumerate(self.labels):
            if i == o1 and a == o1:
                diagonal[k] = 0.5 * ((ea[o1] - ea[o1]) + (ea[o2] - ea[o2]))
            else:
                diagonal[k] = ea[a] - ea[i]
            if i == o2 and a == o2:
                diagonal[k] = np.inf
            elif self.target_multiplicity == 3 and (
                (i == o2 and a == o1) or (i == o1 and a == o2)
            ):
                diagonal[k] = np.inf
        return diagonal

    def ao_components(self, vector: np.ndarray) -> np.ndarray:
        """Build the seven AO response densities of collinear MRSF-TDA."""

        x = self.unpack(self.mask_redundant(vector))
        c = self.ref.coeff
        n = self.ref.nao
        o1, o2 = self.ref.o1, self.ref.o2
        closed = self.ref.closed
        virtual = self.ref.virtual
        s2 = 1.0 / np.sqrt(2.0)

        comp = np.zeros((7, n, n))
        bo2v, bo1v, bco1, bco2, o21v, co12, ball = comp

        if len(virtual):
            bo2v += np.outer(c[:, o2], c[:, virtual] @ x[o2, virtual])
            bo1v += np.outer(c[:, o1], c[:, virtual] @ x[o1, virtual])
            o21v += np.outer(c[:, virtual] @ x[o2, virtual], c[:, o1])
            o21v -= np.outer(c[:, virtual] @ x[o1, virtual], c[:, o2])
        if len(closed):
            left_o1 = c[:, closed] @ x[closed, o1]
            left_o2 = c[:, closed] @ x[closed, o2]
            bco1 += np.outer(left_o1, c[:, o1])
            bco2 += np.outer(left_o2, c[:, o2])
            co12 += np.outer(c[:, o2], left_o1)
            co12 -= np.outer(c[:, o1], left_o2)
            if len(virtual):
                ball += c[:, closed] @ x[np.ix_(closed, virtual)] @ c[:, virtual].T

        ball += bo2v + bo1v + bco1 + bco2
        if self.target_multiplicity == 1:
            ball += x[o2, o1] * np.outer(c[:, o2], c[:, o1])
            ball += x[o1, o2] * np.outer(c[:, o1], c[:, o2])
            ball += x[o1, o1] * s2 * (
                np.outer(c[:, o1], c[:, o1]) - np.outer(c[:, o2], c[:, o2])
            )
        else:
            ball += x[o1, o1] * s2 * (
                np.outer(c[:, o1], c[:, o1]) + np.outer(c[:, o2], c[:, o2])
            )
        return comp

    def mo_action(self, fock_components: np.ndarray) -> np.ndarray:
        """Transform the seven AO Fock-like components back to MRSF space."""

        f = np.asarray(fock_components)
        if f.shape != (7, self.ref.nao, self.ref.nao):
            raise ValueError("Expected seven square AO component matrices")
        c = self.ref.coeff
        o1, o2 = self.ref.o1, self.ref.o2
        closed, virtual = self.ref.closed, self.ref.virtual
        s2 = 1.0 / np.sqrt(2.0)
        ado2v, ado1v, adco1, adco2, ao21v, aco12, agdlr = f
        scr = c.T @ agdlr @ c
        work = scr.copy()

        if len(closed):
            tmp = ado1v @ c[:, o2] + aco12 @ c[:, o1]
            work[closed, o2] += c[:, closed].T @ tmp
            tmp = ado2v @ c[:, o1] - aco12 @ c[:, o2]
            work[closed, o1] += c[:, closed].T @ tmp
        if len(virtual):
            tmp = adco2.T @ c[:, o1] + ao21v.T @ c[:, o2]
            work[o1, virtual] += c[:, virtual].T @ tmp
            tmp = adco1.T @ c[:, o2] - ao21v.T @ c[:, o1]
            work[o2, virtual] += c[:, virtual].T @ tmp

        if self.target_multiplicity == 1:
            work[o1, o1] = (scr[o1, o1] - scr[o2, o2]) * s2
            work[o2, o2] = 0.0
        else:
            work[o1, o1] = (scr[o1, o1] + scr[o2, o2]) * s2
            work[o2, o1] = 0.0
            work[o1, o2] = 0.0
            work[o2, o2] = 0.0
        return self.mask_redundant(self.pack(work))

    def one_electron_action(self, vector: np.ndarray) -> np.ndarray:
        """Apply the alpha/beta ROHF Fock part of the MRSF Hessian."""

        x = self.unpack(self.mask_redundant(vector))
        fa = self.ref.fock_alpha_mo
        fb = self.ref.fock_beta_mo
        work = np.zeros_like(x)
        rows, cols = self.rows, self.cols
        o1, o2 = self.ref.o1, self.ref.o2
        # The spin-adapted O1/O1 amplitude is not an ordinary orbital rotation.
        # It is removed from the generic X F_beta - F_alpha X contraction and
        # reinserted below with the 1/sqrt(2) spin-coupling factors.
        generic = x.copy()
        generic[o1, o1] = 0.0
        generic[o2, o2] = 0.0
        sub = generic[np.ix_(rows, cols)]
        # X F_beta - F_alpha X.  The first term follows the transposed
        # contraction in the original MRSF equations.
        tmp = sub @ fb[np.ix_(cols, cols)].T - fa[np.ix_(rows, rows)] @ sub
        work[np.ix_(rows, cols)] = tmp

        xoo = x[o1, o1]
        s2 = 1.0 / np.sqrt(2.0)
        if self.target_multiplicity == 1:
            for a in cols:
                work[o1, a] += fb[a, o1] * xoo * s2
                work[o2, a] -= fb[a, o2] * xoo * s2
            for i in rows:
                work[i, o1] -= fa[i, o1] * xoo * s2
                work[i, o2] += fa[i, o2] * xoo * s2
            d = (
                -np.dot(fa[o1, rows], generic[rows, o1])
                + np.dot(fa[o2, rows], generic[rows, o2])
                + np.dot(fb[o1, cols], generic[o1, cols])
                - np.dot(fb[o2, cols], generic[o2, cols])
            )
        else:
            for a in cols:
                work[o1, a] += fb[a, o1] * xoo * s2
                work[o2, a] += fb[a, o2] * xoo * s2
            for i in rows:
                work[i, o1] -= fa[i, o1] * xoo * s2
                work[i, o2] -= fa[i, o2] * xoo * s2
            d = (
                -np.dot(fa[o1, rows], generic[rows, o1])
                - np.dot(fa[o2, rows], generic[rows, o2])
                + np.dot(fb[o1, cols], generic[o1, cols])
                + np.dot(fb[o2, cols], generic[o2, cols])
            )
        work[o1, o1] = d * s2 + 0.5 * xoo * (
            fb[o1, o1] + fb[o2, o2] - fa[o1, o1] - fa[o2, o2]
        )
        return self.mask_redundant(self.pack(work))
