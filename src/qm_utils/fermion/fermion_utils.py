import jax
import jax.numpy as jnp
import math
from functools import partial
import numpy as np

from jaxtyping import Array, Int
from numba import njit, prange

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(0, 1))
def get_occupations_bitset(total_bits: int, n_ones: int):
    count = math.comb(total_bits, n_ones)
    init_val = jnp.int64((1 << n_ones) - 1)
    def step_fn(current, _):
        val_to_save = current
        
        c = current & -current
        r = current + c
        next_val = (((r ^ current) >> 2) // c) | r
        
        return next_val.astype(jnp.int64), val_to_save
    _, result = jax.lax.scan(step_fn, init_val, None, length=count)
    return result

@partial(jax.jit, static_argnums=(1,))
def bitset_to_array(state_int: Int[Array, ""], n_modes: int) -> Int[Array, "n_modes"]:
    shifts = jnp.arange(n_modes)
    shifted = jnp.right_shift(state_int, shifts)
    bits = jnp.bitwise_and(shifted, 1)
    return bits.astype(jnp.int8)

@jax.jit
def array_to_bitset(bits: Int[Array, "n_modes"]) -> Int[Array, ""]:
    shifts = jnp.arange(bits.shape[-1])
    return jnp.sum(jnp.left_shift(bits.astype(jnp.int64), shifts), axis=-1)

@partial(jax.jit, static_argnums=(1, 2))
def bitset_to_mode_indices(state_int: int, n_modes: int, max_n_particles: int, fill_value: int=-1) -> jnp.ndarray:
    return jnp.nonzero(
        bitset_to_array(state_int, n_modes), 
        size=max_n_particles, 
        fill_value=fill_value
    )[0]

@njit(fastmath=True, cache=True)
def bitsets_to_array_numba(states, n_modes):
    n_batch = states.shape[0]
    out = np.zeros((n_batch, n_modes), dtype=np.int8)

    for i in range(n_batch):
        val = states[i]
        for b in range(n_modes):
            if (val >> b) & 1:
                out[i, b] = 1
    return out

@njit(parallel=True, fastmath=True, cache=True)
def total_sum_by_table(
    states: np.ndarray,
    sum_table: np.ndarray,
    n_modes: int,
    zero_idx: int
):
    n_states = states.shape[0]
    labels = np.empty(n_states, dtype=np.int64)

    for i in prange(n_states):
        state = states[i]
        current_val = zero_idx
        for mode_idx in range(n_modes):
            if (state >> mode_idx) & 1:
                current_val = sum_table[current_val, mode_idx]
        labels[i] = current_val
    
    return labels
