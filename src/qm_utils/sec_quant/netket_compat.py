
import jax
import jax.numpy as jnp
import netket as nk
from netket.utils import struct

from ..lattice.brillouin_zone import BrillouinZone2D
from .fermions_on_bz import SectorOnBZ, FermionsOnBZ


def sum_indices(initial, indices, sum_table):
    def scan_fn(carry, x):
        new_carry = sum_table[carry, x]
        return new_carry, None
    init_carry = initial
    final_carry, _ = jax.lax.scan(scan_fn, init_carry, indices)
    return final_carry

class SectorConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):

    sector_idx: int = struct.field(pytree_node=False)
    n_particles: int = struct.field(pytree_node=False)
    zero_idx: int = struct.field(pytree_node=False)
    sum_table: jax.Array = struct.field(pytree_node=False)

    def __init__(self, sector_idx, n_particles, zero_idx, sum_table):
        self.sector_idx = sector_idx
        self.n_particles = n_particles
        self.zero_idx = zero_idx
        self.sum_table = sum_table

    def __call__(self, x):
        indices = jnp.nonzero(x, size=self.n_particles)[0]
        total_idx = sum_indices(self.zero_idx, indices, self.sum_table)
        return total_idx == self.sector_idx
    
    def __hash__(self):
        return hash(("SectorConstraint", self.sector_idx, self.n_particles))
    
    def __eq__(self, other):
        if isinstance(other, SectorConstraint):
            return self.sector_idx == other.sector_idx and self.n_particles == other.n_particles
        return False

def get_sector_constraints(bz: BrillouinZone2D, n_particles: int) -> list[SectorConstraint]:
    sum_table = jnp.array(bz.sum_table, dtype=jnp.int32)
    zero_idx = bz.zero()
    sector_constraints = [
        SectorConstraint(
            sector_idx=k_idx,
            n_particles=n_particles,
            zero_idx=zero_idx,
            sum_table=sum_table
        ) for k_idx in range(bz.n_samples)
    ]
    return sector_constraints