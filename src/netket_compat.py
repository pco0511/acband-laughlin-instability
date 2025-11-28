from math import comb
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import netket as nk
from netket.utils import struct
from netket.operator import FermionOperator2ndJax
from einops import rearrange
import tqdm

from .qm_utils.lattice.brillouin_zone import BrillouinZone2D
from .qm_utils.fermion.fermionic_fock import Sector
from .qm_utils.fermion.fermion_utils import bitset_to_array, array_to_bitset


def sum_indices(initial, indices, sum_table):
    def scan_fn(carry, x):
        new_carry = sum_table[carry, x]
        return new_carry, None
    init_carry = initial
    final_carry, _ = jax.lax.scan(scan_fn, init_carry, indices)
    return final_carry

class SectorConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):

    sector_idx: int = struct.field(pytree_node=False)
    max_particles: int = struct.field(pytree_node=False)
    zero_idx: int = struct.field(pytree_node=False)
    sum_table: jax.Array = struct.field(pytree_node=False)

    def __init__(self, sector_idx, max_particles, zero_idx, sum_table):
        self.sector_idx = sector_idx
        self.max_particles = max_particles
        self.zero_idx = zero_idx
        self.sum_table = sum_table
    
    def _apply_single(self, x):
        indices, = jnp.nonzero(x, size=self.max_particles, fill_value=self.zero_idx)
        total_idx = sum_indices(self.zero_idx, indices, self.sum_table)
        return (total_idx == self.sector_idx)

    @jax.jit
    def __call__(self, x):
        if x.ndim == 1:
            return self._apply_single(x)
        else:
            batch_shape = x.shape[:-1]
            x_flat = rearrange(x, "... n -> (...) n")
            results_flat = jax.vmap(self._apply_single)(x_flat)
            return results_flat.reshape(batch_shape)
    
    def __hash__(self):
        return hash(("SectorConstraint", self.sector_idx, self.max_particles))
    
    def __eq__(self, other):
        if isinstance(other, SectorConstraint):
            return self.sector_idx == other.sector_idx and self.max_particles == other.max_particles
        return False

def get_sector_constraints(bz: BrillouinZone2D, max_particles: int) -> list[SectorConstraint]:
    sum_table = jnp.array(bz.sum_table, dtype=jnp.int32)
    zero_idx = bz.zero()
    sector_constraints = [
        SectorConstraint(
            sector_idx=k_idx,
            max_particles=max_particles,
            zero_idx=zero_idx,
            sum_table=sum_table
        ) for k_idx in range(bz.n_samples)
    ]
    return sector_constraints

def get_number_sector_indices(hilb: nk.hilbert.SpinOrbitalFermions) -> list[int]:
    N = hilb.n_orbitals

    assert hilb.n_fermions is None, f"Number of fermions must not be specified: {hilb.n_fermions}"
    
    ends = [0 for _ in range(N + 1)]
    sectors = [np.zeros(comb(N, n_f)) for n_f in range(N + 1)]

    for idx, state in tqdm.tqdm(enumerate(hilb.states()), total=hilb.n_states):
        n_f = int(jnp.sum(state))
        sectors[n_f][ends[n_f]] = idx
        ends[n_f] += 1
    
    assert all(ends[n_f] == comb(N, n_f) for n_f in range(N + 1)), "Sector sizes do not match expected values"

    return sectors

def check_ordering(full_fock: nk.hilbert.SpinOrbitalFermions, number_fixed: nk.hilbert.SpinOrbitalFermions) -> bool:
    states_number_fixed = iter(number_fixed.states())
    for idx, state in enumerate(full_fock.states()):
        if jnp.sum(state) == number_fixed.n_fermions:
            expected_state = next(states_number_fixed)
            if not np.all(state == expected_state):
                print(f"Mismatch at index {idx}:")
                print(f"  Full Fock state: {state}")
                print(f"  Expected state: {expected_state}")
                return False
    assert next(states_number_fixed, None) is None, "Not all states in number-fixed sector were matched"
    return True

def csr_from_nk_fermion_op(
    discrete_op: FermionOperator2ndJax,
    domain_sector: Sector,
    codomain_sector: Sector,
    batch_size=65536
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x = jax.vmap(bitset_to_array, in_axes=(0, None))(
    #     codomain_sector.basis_labels, codomain_sector.full_hilb.n_modes
    # )
    # sections = np.empty(codomain_sector.dim, dtype=np.int32)
    # x_prime, mels = discrete_op.get_conn_flattened(x, sections)
    # bitsets = jax.vmap(array_to_bitset)(x_prime)
    # numbers = np.searchsorted(domain_sector.basis_labels, bitsets)
    # sections1 = np.empty(sections.size + 1, dtype=np.int32)
    # sections1[1:] = sections
    # sections1[0] = 0
    # return scipy.sparse.csr_matrix(
    #     (mels, numbers, sections1), 
    #     shape=(codomain_sector.dim, domain_sector.dim)
    # )

    vectorized_bitset_to_array = jax.vmap(bitset_to_array, in_axes=(0, None))
    vectorized_array_to_bitset = jax.vmap(array_to_bitset, in_axes=(0,))
    mels = []
    indices = []
    indptr = np.empty(codomain_sector.dim + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(0, codomain_sector.dim, batch_size):
        x_batch = vectorized_bitset_to_array(
            codomain_sector.basis_labels[i:i+batch_size], codomain_sector.full_hilb.n_modes
        )
        sections = indptr[i + 1:i + 1 + batch_size]
        x_prime_batch, mels_batch = discrete_op.get_conn_flattened(x_batch, sections)
        x_int_batch = vectorized_array_to_bitset(x_prime_batch)
        indices_batch = np.searchsorted(domain_sector.basis_labels, x_int_batch)
        sections += indptr[i]
        mels.append(mels_batch)
        indices.append(indices_batch)
    
    mels = np.concatenate(mels)
    indices = np.concatenate(indices)

    return scipy.sparse.csr_matrix(
        (mels, indices, indptr), 
        shape=(codomain_sector.dim, domain_sector.dim)
    )

    