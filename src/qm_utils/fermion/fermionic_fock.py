from collections import defaultdict
from typing import Any, Callable
from math import comb

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxtyping import Array, Complex, Float, Int, Bool 


from .fermion_utils import (
    get_occupations_bitset,
    bitset_to_array,
    bitset_to_mode_indices,
)

from ..config import LARGE_BATCH_SIZE

from tqdm.auto import tqdm

class DiscreteFermionicFockSpace:
    def __init__(
        self,
        mode_labels: list,
        particle_numbers: None | int | list[int]
    ):
        self.mode_labels = list(mode_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.mode_labels)}
        self.n_modes = len(self.mode_labels)

        if self.n_modes > 64:
            raise ValueError("Fock space with more than 64 modes is not supported")

        if particle_numbers is None:
            self.particle_numbers = list(range(self.n_modes + 1))
        elif isinstance(particle_numbers, int):
            self.particle_numbers = [particle_numbers]
        else:
            self.particle_numbers = sorted(particle_numbers)

        self.states = np.sort(np.concatenate(
            [
                np.array(get_occupations_bitset(self.n_modes, n_f))
                for n_f in self.particle_numbers
            ]
        ))
        # self.number_sector_sizes = np.array([
        #     comb(self.n_modes, n_f) for n_f in self.particle_numbers
        # ])
        # size_cumsum = np.cumsum(self.number_sector_sizes)
        # self.number_sector_starts = np.concatenate((
        #     [0],
        #     size_cumsum[:-1]
        # ))
        # self.number_sector_ends = size_cumsum

        self.dim = len(self.states)
        self._state_to_index = None

    def __repr__(self) -> str:
        return (
            f"DiscreteFermionicFockSpace(n_modes={self.n_modes}, "
            f"particle_numbers={self.particle_numbers}, dim={self.dim})"
        )

    @property
    def state_to_index(self) -> dict[int, int]:
        if self._state_to_index is None:
            self._state_to_index = {
                state: idx for idx, state in enumerate(self.states)
            }
        return self._state_to_index

    @property
    def dense_operator_size(self) -> int:
        return ((self.dim) ** 2) * 16

    def decompose_sector_by_labels(self, state_labels, sector_labels):
        sector_dict = {}
        for sector_idx, sector_label in enumerate(sector_labels):
            basis_indices = np.where(state_labels == sector_idx)[0]
            sector_dict[sector_label] = Sector(
                full_hilb=self,
                sector_label=sector_label,
                basis_labels=self.states[basis_indices],
            )
        return sector_dict
    
    def decompose_sectors(self, labeling_fn:Callable[[int], int], sector_labels: list, pbar=False):
        batched_labeling = jax.jit(jax.vmap(labeling_fn))
        labels = np.empty(self.dim, dtype=int)
        
        if pbar:
            batch_slices = tqdm(list(range(0, self.dim, LARGE_BATCH_SIZE)), desc="labeling sectors")
        else:
            batch_slices = range(0, self.dim, LARGE_BATCH_SIZE)

        for i in batch_slices:
            state_batch = self.states[i:i+LARGE_BATCH_SIZE]
            labels[i:i+LARGE_BATCH_SIZE] = batched_labeling(state_batch)

        return self.decompose_sector_by_labels(labels, sector_labels)
    

    # def number_sectors(self) -> int:
    #     pass


class Sector:
    def __init__(
        self,
        full_hilb: DiscreteFermionicFockSpace,
        sector_label: tuple[float, float],
        basis_labels: np.ndarray
    ):
        self.full_hilb = full_hilb
        self.sector_label = sector_label
        self.basis_labels = basis_labels
        self.dim = len(basis_labels)
        self._label_to_index = None

    def __repr__(self) -> str:
        return (
            f"Sector(label={self.sector_label}, dim={self.dim}, "
            f"full_hilb_n_modes={self.full_hilb.n_modes})"
        )

    @property
    def label_to_index(self) -> dict[int, int]:
        if self._label_to_index is None:
            self._label_to_index = {
                label: idx for idx, label in enumerate(self.basis_labels)
            }
        return self._label_to_index
