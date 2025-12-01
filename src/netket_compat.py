from math import comb
import numpy as np
import scipy
import jax
jax.config.update("jax_enable_x64", True)
import logging
logger = logging.getLogger("jax._src.xla_bridge")
logger.setLevel(logging.ERROR)
from threadpoolctl import threadpool_limits


import jax.numpy as jnp
import netket as nk
from netket.utils import struct
from netket.operator import FermionOperator2ndJax
from einops import rearrange
from tqdm.auto import tqdm

from numba import njit, prange
from joblib import Parallel, delayed

from .qm_utils.lattice.brillouin_zone import BrillouinZone2D
from .qm_utils.fermion.fermionic_fock import Sector
from .qm_utils.fermion.fermion_utils import bitset_to_array, array_to_bitset, bitsets_to_array_numba


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

@njit(fastmath=True, cache=True)
def _numba_array_to_index_search(x_prime_batch, basis_labels):
    n_items, n_modes = x_prime_batch.shape
    indices = np.empty(n_items, dtype=np.int64)

    for i in range(n_items):
        val = 0
        for b in range(n_modes):
            if x_prime_batch[i, b] != 0:
                val += (1 << b)
        idx = np.searchsorted(basis_labels, val)
        indices[i] = idx
    
    return indices

def _process_batch(
    start_idx: int,
    batch_size: int,
    codomain_labels: np.ndarray, # Read-only, shared memory efficient on Linux
    domain_labels: np.ndarray,   # Read-only
    discrete_op,                 # The Operator Object
    n_modes: int
):
    batch_labels = codomain_labels[start_idx : start_idx + batch_size]
    actual_batch_size = len(batch_labels)
    x_batch = bitsets_to_array_numba(batch_labels, n_modes)
    local_sections = np.empty(actual_batch_size, dtype=np.int32)
    x_prime_batch, mels_batch = discrete_op.get_conn_flattened(x_batch, local_sections)
    indices_batch = _numba_array_to_index_search(x_prime_batch, domain_labels)
    row_nnz = np.diff(local_sections, prepend=0)
    return mels_batch, indices_batch, row_nnz

def csr_from_nk_fermion_op(
    discrete_op: FermionOperator2ndJax,
    domain_sector: Sector,
    codomain_sector: Sector,
    batch_size: int=4096,
    n_jobs: int=-1,
    pbar: bool = False
) -> scipy.sparse.csr_matrix:
    n_total = codomain_sector.dim
    starts = range(0, n_total, batch_size)

    if pbar:
        task_iterator = tqdm(starts, desc="Building CSR matrix")
    else:
        task_iterator = starts

    with threadpool_limits(limits=1, user_api="blas"):
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_process_batch)(
                start,
                batch_size,
                codomain_sector.basis_labels,
                domain_sector.basis_labels,
                discrete_op,
                codomain_sector.full_hilb.n_modes
            )
            for start in task_iterator
        )

        all_mels = []
        all_indices = []
        all_row_nnz = []
        for mels_batch, indices_batch, row_nnz in results:
            all_mels.append(mels_batch)
            all_indices.append(indices_batch)
            all_row_nnz.append(row_nnz)

    data = np.concatenate(all_mels)
    indices = np.concatenate(all_indices)
    row_nnz = np.concatenate(all_row_nnz)
    
    indptr = np.zeros(n_total + 1, dtype=np.int32)
    np.cumsum(row_nnz, out=indptr[1:])

    return scipy.sparse.csr_matrix(
        (data, indices, indptr),
        shape=(codomain_sector.dim, domain_sector.dim)
    )


    # vectorized_bitset_to_array = jax.vmap(bitset_to_array, in_axes=(0, None))
    # vectorized_array_to_bitset = jax.vmap(array_to_bitset, in_axes=(0,))
    # mels = []
    # indices = []
    # indptr = np.empty(codomain_sector.dim + 1, dtype=np.int32)
    # indptr[0] = 0
    # nnz = 0
    # real_nnz = 0

    # if pbar:
    #     iterator = tqdm.trange(0, codomain_sector.dim, batch_size)
    # else:
    #     iterator = range(0, codomain_sector.dim, batch_size)

    # for i in iterator:
    #     x_batch = vectorized_bitset_to_array(
    #         codomain_sector.basis_labels[i:i+batch_size], codomain_sector.full_hilb.n_modes
    #     )
    #     sections = indptr[i + 1:i + 1 + batch_size]
    #     x_prime_batch, mels_batch = discrete_op.get_conn_flattened(x_batch, sections)
    #     x_int_batch = vectorized_array_to_bitset(x_prime_batch)
    #     indices_batch = np.searchsorted(domain_sector.basis_labels, x_int_batch)
    #     mels.append(mels_batch)
    #     indices.append(indices_batch)
    #     nnz += mels_batch.shape[0]
    #     real_nnz += np.sum(np.abs(mels_batch) > 1e-12)
    #     if pbar:
    #         iterator.set_postfix(nnz=nnz, real_nnz=real_nnz)
    
    # for i in range(0, codomain_sector.dim, batch_size):
    #     indptr[i + 1:i + 1 + batch_size] += indptr[i]

    # mels = np.concatenate(mels)
    # indices = np.concatenate(indices)

    # return scipy.sparse.csr_matrix(
    #     (mels, indices, indptr), 
    #     shape=(codomain_sector.dim, domain_sector.dim)
    # )

    