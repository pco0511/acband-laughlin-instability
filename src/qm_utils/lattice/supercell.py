import itertools
import random

import numpy as np

from einops import rearrange, pack

from .lattice import Lattice2D

__all__ = [
    "SuperCell2D"
]

class SuperCell2D:
    def __init__(
        self,
        lattice: Lattice2D,
        sublattice: Lattice2D,
        eps: float=1e-12
    ):
        self.lattice = lattice
        self.sublattice = sublattice

        # L1 = m11 a1 + m12 a2
        # L2 = m21 a1 + m22 a2
        a1, a2 = self.lattice.lattice_vectors
        L1, L2 = self.sublattice.lattice_vectors

        (m11, m12), offset1 = lattice.divmod(L1)
        (m21, m22), offset2 = lattice.divmod(L2)

        if np.linalg.norm(offset1) >= eps or np.linalg.norm(offset2) >= eps:
            raise ValueError("Sublattice is not a sublattice of given lattice.")
        
        self.a1 = a1
        self.a2 = a2
        self.L1 = L1
        self.L2 = L2

        self.m11 = m11
        self.m12 = m12
        self.m21 = m21
        self.m22 = m22

        A = max(np.linalg.norm(a1), np.linalg.norm(a2))
        L = min(np.linalg.norm(L1), np.linalg.norm(L2))
        M = np.ceil(L / A).astype(int)

        xx = np.linspace(-M, M, 2 * M + 1)
        yy = np.linspace(-M, M, 2 * M + 1)
        mgrid = np.meshgrid(xx, yy)
        candidate_coords = pack(mgrid, "x y *")
        candidates = lattice.get_points(*mgrid, flatten=False)

        candidates = rearrange(candidates, "x y a -> (x y) a")
        candidate_coords = rearrange(candidate_coords, "x y a -> (x y) a")

        coords, _ = sublattice.divmod(candidates)
        wigner_seitz_mask = np.logical_and(coords[:, 0]==0, coords[:, 1]==0)
        
        self.points = candidates[wigner_seitz_mask]
        self.coords = candidate_coords[wigner_seitz_mask]
        self.n_points = np.sum(wigner_seitz_mask)

        # maps
        self.idx_from_coord = {(m1, m2) : idx for idx, (m1, m2) in enumerate(self.coords)}

        # look up tables
        N_p = self.n_points
        self.neg_table = np.zeros((N_p,), dtype=np.int64)
        self.sum_table = np.zeros((N_p, N_p), dtype=np.int64)
        self.sub_table = np.zeros((N_p, N_p), dtype=np.int64)

        # initialization
        for i in range(N_p):
            self.neg_table[i] = self._idx_neg(i)

        for i, j in itertools.product(range(N_p), repeat=2):
            self.sum_table[i, j] = self._idx_sum(i, j)

        for i, j in itertools.product(range(N_p), repeat=2):
            j_neg = self.neg_table[j]
            self.sub_table[i, j] = self.sum_table[i, j_neg]
        
    @property
    def N_p(self): 
        """alias for n_points"""
        return self.n_points
    
    def __iter__(self):
        for idx in range(self.n_points):
            yield self.points[idx]

    def __getitem__(self, idx):
        return self.points[idx]

    def fold_coord(self, coord):
        pos = self.lattice.pos_from_coord(coord)
        g, _ = self.sublattice.divmod(pos)
        # G = d1 L1 + d2 L2 = (d1 m11 + d2 m21) a1 + (d1 m12 + d2 m22) a2
        d1, d2 = g
        m1, m2 = coord
        new1 = m1 - (d1 * self.m11 + d2 * self.m21)
        new2 = m2 - (d1 * self.m12 + d2 * self.m22)
        return (new1, new2), g

    def _idx_neg(self, idx):
        coord = self.coords[idx]
        neg_coord = tuple(-x for x in coord)
        folded_coord, _ = self.fold_coord(neg_coord)
        return self.idx_from_coord[folded_coord]

    def _idx_sum(self, idx1, idx2):
        coord1 = self.coords[idx1]
        coord2 = self.coords[idx2]
        sum_coord = tuple(x + y for x, y in zip(coord1, coord2))
        folded_coord, _ = self.fold_coord(sum_coord)
        return self.idx_from_coord[folded_coord]
    
    def zero(self):
        return self.idx_from_coord[(0, 0)]

    def neg(self, idx):
        return self.neg_table[idx]
    
    def sum(self, idx1, idx2):
        return self.sum_table[idx1, idx2]
    
    def sub(self, idx1, idx2):
        return self.sub_table[idx1, idx2]