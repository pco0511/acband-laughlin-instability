import itertools
import functools

import numpy as np

from ..lattice.brillouin_zone import BrillouinZone2D

class SectorOnBZ:
    def __init__(self, sector_label: tuple[float, float], bz: BrillouinZone2D, states: np.ndarray):
        self.sector_label = sector_label
        self.bz = bz
        self.states = states
        self.idx_from_state = {int(state): idx for idx, state in enumerate(self.states)}

class FermionsOnBZ:
    def __init__(self, bz: BrillouinZone2D, n_particles: int):
        self.bz = bz
        self.n_particles = n_particles

    def bitset_from_indices(self, indices):
        bitset = 0
        for idx in indices:
            bitset |= (1 << idx)
        return bitset

    def get_momentum_sectors(self):
        if self.n_particles > self.bz.n_samples:
            raise ValueError("cannot have more particles than sites in BZ")
        if self.bz.n_samples > 64:
            raise ValueError("momentum sector enumeration not supported for n_samples > 64")
    
        momentum_sectors: dict[int, list[int]] = dict()
        for comb in itertools.combinations(range(self.bz.n_samples), self.n_particles):
            total_momentum_idx = functools.reduce(self.bz.sum, comb)
            
            bitset = self.bitset_from_indices(comb)
            if total_momentum_idx not in momentum_sectors:
                momentum_sectors[total_momentum_idx] = []
            momentum_sectors[total_momentum_idx].append(bitset)
        
        return {
            k_idx: SectorOnBZ(
                sector_label=self.bz[k_idx],
                bz=self.bz,
                states=np.array(v_states, dtype=np.uint64)
            ) for k_idx, v_states in momentum_sectors.items()
        }
    
