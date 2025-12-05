import time
import itertools
from typing import Callable

import numpy as np
import scipy
from scipy.signal import correlate

from tqdm.auto import tqdm, trange

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .qm_utils.lattice.lattice import Lattice2D
from .qm_utils.lattice.brillouin_zone import BrillouinZone2D

from einops import einsum, rearrange, pack

from numba import njit, prange

def eta_factors(
        g_coords: np.ndarray
    ) -> np.ndarray:
    m, n = rearrange(g_coords, "... d -> d ...")
    return (-1) ** ((n + m + m * n) % 2)
    
def K_func1(
        x: np.ndarray,
        args: tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
    K, b1, b2, b3 = args
    # coeff = -np.sqrt(3) / (4 * np.pi) * K
    coeff = -np.sqrt(3) / (2 * np.pi) * K # ????? 4 pi -> 2 pi
    cos1 = np.cos(einsum(x, b1, "... d, d -> ..."))
    cos2 = np.cos(einsum(x, b2, "... d, d -> ..."))
    cos3 = np.cos(einsum(x, b3, "... d, d -> ..."))
    return coeff * (cos1 + cos2 + cos3)

def K_func2(
        x: np.ndarray,
        args: tuple[int, float, float, np.ndarray, np.ndarray]
    ) -> np.ndarray:
    # solenoid
    m, sigma, R, a1, a2 = args

    N = int(np.ceil(2 * R))
    ns = np.linspace(-N, N, 2 * N + 1)
    n1 = ns[:, None, None]
    n2 = ns[None, :, None]
    lattice_points = n1 * a1[None, None, :] + n2 * a2[None, None, :]
    flatten_lattice_points = rearrange(lattice_points, "n1 n2 d -> (n1 n2) d")
    flatten_lattice_points_norms = np.linalg.norm(flatten_lattice_points, axis=-1)
    flatten_lattice_points = flatten_lattice_points[flatten_lattice_points_norms < R, :]
    
    x_batch_shape = x.shape[:-1]
    x_flatten = rearrange(x, "... d -> (...) d")
    
    origin_point = np.array([[0.0, 0.0]])
    x_calc = np.concatenate([x_flatten, origin_point], axis=0)
    
    k_vals_calc = np.zeros(x_calc.shape[0])
    batch_size = 64
    num_points = flatten_lattice_points.shape[0]
    
    for i in range(0, num_points, batch_size):
        batch_lattice_points = flatten_lattice_points[i:i+batch_size]
        
        diffs = x_calc[:, None, :] - batch_lattice_points[None, :, :]
        dists_square = np.sum(diffs ** 2, axis=-1) / (sigma ** 2)
        dists = np.sqrt(dists_square) 

        with np.errstate(divide='ignore', invalid='ignore'):
            chis = np.where(dists < 1, (dists_square - 1) / 2, np.log(dists))
        
        k_vals_calc += np.sum(chis, axis=1)
    
    n = 1 / (np.abs(a1[0] * a2[1] - a1[1] * a2[0]))
    c1 = (np.pi / 2) * n
    c0 = np.pi * n * (R ** 2) * (np.log(R / sigma) - 0.5)
    
    background = c1 * np.sum(x_calc ** 2, axis=-1) + c0
    k_vals_calc -= background

    k_vals_calc *= m
    
    offset = k_vals_calc[-1]
    k_vals_calc -= offset
    
    k_vals_final = k_vals_calc[:-1]

    k_vals = np.reshape(k_vals_final, shape=x_batch_shape)
    return k_vals

def wg_fourier_components(
    K_func,
    lattice: Lattice2D, 
    resolution: int,
    *,
    flatten: bool
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, resolution, endpoint=False)
    y = np.linspace(0, 1, resolution, endpoint=False)
    
    mgrid = np.meshgrid(x, y, indexing='ij', sparse=True)
    samples = lattice.get_points(*mgrid, flatten=False)
    K_vals = K_func(samples)
    
    wg_raw = np.fft.ifft2(np.exp(-2 * K_vals))
    wg = np.fft.fftshift(wg_raw)
    
    m_vals = np.fft.fftshift(np.fft.fftfreq(resolution) * resolution).astype(int)
    m1_grid, m2_grid = np.meshgrid(m_vals, m_vals, indexing='ij')
    g_coords = np.stack((m1_grid, m2_grid), axis=-1)
    
    if flatten:
        g_coords = rearrange(g_coords, "m n d -> (m n) d")
        wg = rearrange(wg, "m n -> (m n)")

    return g_coords, wg
    
def cross_z(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return arr1[..., 0] * arr2[..., 1] - arr1[..., 1] * arr2[..., 0]

def norm_square(arr: np.array, axis: int) -> np.ndarray:
    return np.sum(np.abs(arr) ** 2, axis=axis)

def acband_normalization_constants(
    bz: BrillouinZone2D,
    lB: float,
    g_coords: np.ndarray, # shape: (res, res, 2)
    wg: np.ndarray, # shape: (res, res),
    eps: float
) -> np.ndarray:
    eta_vals = eta_factors(g_coords) # shape: (res, res)
    recip_lattice = bz.reciprocal_lattice
    gs = recip_lattice.get_points(g_coords)  # shape: (res, res, 2)
    ks = bz.k_points  # shape: (N_s, 2)
    k_cross_g = cross_z(ks[:, None, None, :], gs[None, ...])  # shape: (N_s, res, res)
    g_squared = norm_square(gs, axis=-1)  # shape: (res, res)
    
    lB2 = lB ** 2
    k_indeps = (wg * eta_vals * np.exp(-g_squared * lB2 / 4))  # shape: (res, res)
    k_deps = np.exp(1j * lB2 * k_cross_g) # shape: (N_s, res, res)
    norm_invs2 = einsum(k_indeps, k_deps, "m n, k m n -> k")  # shape: (N_s,)
    
    if np.any(norm_invs2 < 0):
        norm_invs2 *= -1

    assert np.all(np.abs(np.imag(norm_invs2)) < eps), f"Normalization constants have significant imaginary parts"
    assert np.all(np.real(norm_invs2) > eps), f"Normalization constants have non-positive real parts: {np.min(np.real(-norm_invs2))}"
    
    return 1.0 / np.sqrt(np.abs(norm_invs2))

@njit(parallel=True, fastmath=True, cache=True)
def _compute_LLL_numba(
    ks: np.ndarray,       # (N_s, 2)
    gs: np.ndarray,       # (N_g, 2)
    eta_vals: np.ndarray, # (N_g,)
    lB: float
):
    N_s = ks.shape[0]
    N_g = gs.shape[0]

    total_tasks = N_s * N_s

    ff_out = np.zeros((N_s, N_s, N_g), dtype=np.complex128)

    lB2 = lB * lB
    lB2_half = lB2 / 2.0
    lB2_quarter_neg = -lB2 / 4.0

    for idx in prange(total_tasks):
        i = idx // N_s
        j = idx % N_s

        k1_x = ks[i, 0]
        k1_y = ks[i, 1]
        k2_x = ks[j, 0]
        k2_y = ks[j, 1]

        cross_k1_k2 = k1_x * k2_y - k1_y * k2_x
        exp_imag_2 = lB2_half * cross_k1_k2
        
        kp_x = k1_x + k2_x
        kp_y = k1_y + k2_y
        
        km_x = k1_x - k2_x
        km_y = k1_y - k2_y

        for m in range(N_g):
            g_x = gs[m, 0]
            g_y = gs[m, 1]
            
            # cross_z(k1 + k2, g)
            cross_kp_g = kp_x * g_y - kp_y * g_x
            exp_imag_1 = lB2_half * cross_kp_g
            
            # norm_sq(k1 - k2 - g)
            dx = km_x - g_x
            dy = km_y - g_y
            norm_sq = dx * dx + dy * dy
            exp_real = lB2_quarter_neg * norm_sq
            
            total_phase = exp_imag_1 + exp_imag_2
            
            # exp(real + 1j * phase)
            term = np.exp(exp_real + 1j * total_phase)
            
            ff_out[i, j, m] = eta_vals[m] * term
                
    return ff_out


def LLL_form_factors(
    bz: BrillouinZone2D,
    lB: float,
    g_coords: np.ndarray, # shape: (..., 2)
) -> np.ndarray:
    g_batch_shape = g_coords.shape[:-1]
    g_coords_flatten = rearrange(g_coords, "... d -> (...) d")
    ks = bz.k_points  # shape: (N_s, 2)
    gs = bz.reciprocal_lattice.get_points(g_coords_flatten) # (N_g, 2)
    eta_vals = eta_factors(g_coords_flatten)  # shape: (N_g,)
    ff_flatten = _compute_LLL_numba(ks, gs, eta_vals, lB)
    ff_k1_k2_g = ff_flatten.reshape(bz.N_s, bz.N_s, *g_batch_shape)  # shape: (N_s, N_s, ...)
    return ff_k1_k2_g

@njit(parallel=True, fastmath=True, cache=True)
def _compute_lambda(
    G_coords: np.ndarray,       # (N_grid, N_grid, 2) integers
    ff_k_p_g: np.ndarray,       # (N_s, N_s, res, res) complex128
    wg_window: np.ndarray,      # (window_size, window_size) - pre-sliced wg
    normalizations: np.ndarray, # (N_s,)
    G_radius: int
):
    N_grid = G_coords.shape[0]
    N_s = ff_k_p_g.shape[0]
    
    Lambda_k_p_G = np.zeros((N_grid, N_grid, N_s, N_s), dtype=np.complex128)
    
    win_h = wg_window.shape[0]
    win_w = wg_window.shape[1]

    total_tasks = N_grid * N_grid

    for task_idx in prange(total_tasks):
        i = task_idx // N_grid
        j = task_idx % N_grid

        m = G_coords[i, j, 0]
        n = G_coords[i, j, 1]
        
        start1 = G_radius + m
        start2 = G_radius + n
        
        for k in range(N_s):
            for p in range(N_s):
                val = 0.0 + 0.0j
                for x in range(win_h):
                    ff_x = start1 + x
                    for y in range(win_w):
                        ff_y = start2 + y
                        val += ff_k_p_g[k, p, ff_x, ff_y] * wg_window[x, y]
                Lambda_k_p_G[i, j, k, p] = normalizations[k] * normalizations[p] * val
                    
    return Lambda_k_p_G

def _compute_row_old(
    k_idx, 
    ff_row_k,       # (N_s, res, res) - k번째 행 전체
    wg_window,      # (win, win)
    norm_k, 
    all_norm_p      # (N_s,)
):
    N_s = ff_row_k.shape[0]
    res_h = ff_row_k.shape[1] - wg_window.shape[0] + 1
    res_w = ff_row_k.shape[2] - wg_window.shape[1] + 1
    
    row_results = np.empty((N_s, res_h, res_w), dtype=np.complex128)
    
    for p in range(N_s):
        val = correlate(ff_row_k[p], wg_window, mode='valid', method='fft')
        row_results[p] = val * (norm_k * all_norm_p[p])
        
    return k_idx, row_results


def _compute_lambda_old(
    G_coords: np.ndarray,       # (N_grid, N_grid, 2) integers
    ff_k_p_g: np.ndarray,       # (N_s, N_s, res, res) complex128
    wg_window: np.ndarray,      # (window_size, window_size) - pre-sliced wg
    normalizations: np.ndarray, # (N_s,)
    G_radius: int
):
    N_s = ff_k_p_g.shape[0]
    
    h_out = ff_k_p_g.shape[2] - wg_window.shape[0] + 1
    w_out = ff_k_p_g.shape[3] - wg_window.shape[1] + 1
    
    N_grid = G_coords.shape[0]
    assert h_out == N_grid, f"Output shape mismatch: {h_out} vs {N_grid}"

    with threadpool_limits(limits=8, user_api='blas'):
        results = Parallel(n_jobs=16)(
            delayed(_compute_row_old)(
                k, 
                ff_k_p_g[k],
                wg_window, 
                normalizations[k], 
                normalizations
            )
            for k in range(N_s)
        )

    Lambda_k_p_G = np.zeros((N_grid, N_grid, N_s, N_s), dtype=np.complex128)

    for k, row_res in results:
        Lambda_k_p_G[:, :, k, :] = row_res.transpose(1, 2, 0)

    return Lambda_k_p_G


def _compute_row_batch_fft(
    k_idx, 
    ff_row_k_batch, # (N_s, res_h, res_w)
    wg_fft_conj,    # (fft_h, fft_w)
    norm_k_val, 
    all_norm_p,     # (N_s,)
    valid_h, 
    valid_w
):
    input_fft = scipy.fft.fft2(
        ff_row_k_batch, 
        s=wg_fft_conj.shape,
        axes=(-2, -1), 
        workers=1
    )
    input_fft *= wg_fft_conj
    correlated_batch = scipy.fft.ifft2(input_fft, axes=(-2, -1), workers=1)
    valid_result = correlated_batch[:, :valid_h, :valid_w]

    factors = norm_k_val * all_norm_p
    valid_result *= factors[:, None, None]
    
    return k_idx, valid_result

def _compute_lambda_fft(
    G_coords: np.ndarray,       
    ff_k_p_g: np.ndarray,       # (N_s, N_s, 128, 128)
    wg_window: np.ndarray,      # (96, 96)
    normalizations: np.ndarray, 
    G_radius: int
):
    N_s = ff_k_p_g.shape[0]
    res_h, res_w = ff_k_p_g.shape[2], ff_k_p_g.shape[3]
    win_h, win_w = wg_window.shape
    
    h_out = res_h - win_h + 1
    w_out = res_w - win_w + 1
    
    N_grid = G_coords.shape[0]
    assert h_out == N_grid, f"Output shape mismatch: {h_out} vs {N_grid}"

    fft_h = scipy.fft.next_fast_len(res_h + win_h - 1)
    fft_w = scipy.fft.next_fast_len(res_w + win_w - 1)
    
    wg_padded = np.zeros((fft_h, fft_w), dtype=np.complex128)
    wg_padded[:win_h, :win_w] = wg_window
    
    wg_fft_conj = np.conj(scipy.fft.fft2(wg_padded))
    
    with threadpool_limits(limits=1, user_api='blas'):
        results = Parallel(n_jobs=32, prefer="threads")(
            delayed(_compute_row_batch_fft)(
                k, 
                ff_k_p_g[k],
                wg_fft_conj,
                normalizations[k], 
                normalizations,
                h_out,
                w_out
            )
            for k in range(N_s)
        )
    
    Lambda_k_p_G = np.zeros((N_grid, N_grid, N_s, N_s), dtype=np.complex128)

    for k, row_res in results:
        Lambda_k_p_G[:, :, k, :] = row_res.transpose(1, 2, 0)

    return Lambda_k_p_G

def acband_form_factors(
    bz: BrillouinZone2D,
    lB: float,
    K_func: Callable[[np.ndarray], np.ndarray],
    fourier_resolution: int, # 2^n is preferred
    G_radius: int,
    *,
    fft: bool=False,
    pbar: bool=None,
    eps: float=1e-10
) -> tuple[np.ndarray, np.ndarray]:
    if pbar is not None:
        print("pbar argument is removed")
    # compute \Lambda^{k, p}_G
    g_coords, wg = wg_fourier_components(
        K_func, bz.lattice, fourier_resolution, flatten=False
    )
    # shape: (res, res, 2), (res, res)

    normalizations = acband_normalization_constants(
        bz, lB, g_coords, wg, eps=eps
    ) # shape: (N_s,)
    
    ff_k_p_g = LLL_form_factors(
        bz, lB, g_coords
    ) # shape: (N_s, N_s, res, res)
    
    N_grid = 2 * G_radius + 1
    G_coords = np.indices((N_grid, N_grid)).transpose(1, 2, 0) - G_radius
    
    wg_window = wg[G_radius:-G_radius, G_radius:-G_radius]
    if fft:
        Lambda_k_p_G = _compute_lambda_fft(
            G_coords, 
            ff_k_p_g, 
            wg_window, 
            normalizations, 
            G_radius
        ) # shape: (N_grid, N_grid, N_s, N_s)
    else:
        Lambda_k_p_G = _compute_lambda(
            G_coords, 
            ff_k_p_g, 
            wg_window, 
            normalizations, 
            G_radius
        ) # shape: (N_grid, N_grid, N_s, N_s)

    return G_coords, Lambda_k_p_G

def interaction_matrix(
    bz: BrillouinZone2D,
    G_coords: np.ndarray,
    ac_ff: np.ndarray,
    V: Callable[[np.ndarray], np.ndarray],
):
    G_vecs = bz.reciprocal_lattice.get_points(G_coords)
    start_idx = 2
    end_idx = G_coords.shape[0] - 2
    G_vecs_center = G_vecs[start_idx:end_idx, start_idx:end_idx]

    N_s = bz.n_samples
    A = bz.lattice.unit_cell_area * N_s
    # k, p, q
    # (k, p) -> (k + q, p - q)
    int_mat = np.zeros((N_s, N_s, N_s), dtype=np.complex128)
    for k, p, q in itertools.product(range(N_s), repeat=3):
        k1 = bz.sum(k, q) # k + q
        k2 = bz.sub(p, q) # p - q
        k3 = p
        k4 = k

        # k1_coord = bz.k_coords[k1]
        # k2_coord = bz.k_coords[k2]
        k3_coord = bz.k_coords[k3]
        k4_coord = bz.k_coords[k4]
        q_coord = bz.k_coords[q]
        q_vec = bz.k_points[q]

        _, G1_coord = bz.fold_coord(k4_coord + q_coord) # k1 + G1 = k4 + q
        _, G2_coord = bz.fold_coord(k3_coord - q_coord) # k2 + G2 = k3 - q

        q_vec_shited_grid = G_vecs_center + q_vec
        V_grid = V(q_vec_shited_grid)

        # for Lambda 1:
        l1_start_x = start_idx - G1_coord[0]
        l1_end_x   = end_idx   - G1_coord[0]
        l1_start_y = start_idx - G1_coord[1]
        l1_end_y   = end_idx   - G1_coord[1]

        # for Lambda 2:
        l2_start_x = start_idx - G2_coord[0]
        l2_end_x   = end_idx   - G2_coord[0]
        l2_start_y = start_idx - G2_coord[1]
        l2_end_y   = end_idx   - G2_coord[1]

        Lambda_1_grid = ac_ff[l1_start_x:l1_end_x, l1_start_y:l1_end_y, k1, k4][::-1, ::-1]
        Lambda_2_grid = ac_ff[l2_start_x:l2_end_x, l2_start_y:l2_end_y, k2, k3]

        interactions = V_grid * Lambda_1_grid * Lambda_2_grid
        coeff = (1 / (2 * A)) * np.sum(interactions)
        int_mat[k, p, q] = coeff

    return int_mat

def interaction_hamiltonian_terms(
    bz: BrillouinZone2D,
    int_mat: np.ndarray,
):
    terms = []
    weights = []
    N_s = bz.n_samples
    for k, p, q in itertools.product(range(N_s), repeat=3):
        k1 = bz.sum(k, q) # k + q
        k2 = bz.sub(p, q) # p - q # p - q
        k3 = p
        k4 = k
        weights.append(
            complex(int_mat[k, p, q])
        )
        terms.append(
            ((k1, 1), (k2, 1), (k3, 0), (k4, 0))
        )
    return terms, weights

def hole_hartree_self_energies(
    bz: BrillouinZone2D,
    G_vecs: np.ndarray,
    ac_ff: np.ndarray,
    V: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    N_s = bz.n_samples
    A = bz.lattice.unit_cell_area * N_s

    V_G = V(G_vecs) # shape: (N_grid, N_grid)
    k_indices = np.arange(N_s)
    rho_G = ac_ff[:, :, k_indices, k_indices] # taking diagonal elements
    sum_rho_minus_G_k_prime = np.sum(rho_G, axis=2)[::-1, ::-1] # shape: (N_grid, N_grid)
    k_indeps = V_G * sum_rho_minus_G_k_prime # shape: (N_grid, N_grid)
    hartree_energies = (1 / A) * einsum(
        k_indeps, rho_G, "m n, m n k -> k"
    )

    assert np.all(np.abs(np.imag(hartree_energies)) < 1e-6), f"Hartree energies have significant imaginary parts: {np.max(np.abs(np.imag(hartree_energies)))}"

    return np.real(hartree_energies) # shape: (N_s,)

def hole_fock_self_energies(
    bz: BrillouinZone2D,
    G_vecs: np.ndarray,
    ac_ff: np.ndarray,
    V: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    N_s = bz.n_samples
    A = bz.lattice.unit_cell_area * N_s    
    k_vecs = bz.k_points # shape: (N_s, 2)
    fock_energies = np.zeros((N_s,), dtype=np.complex128)
    for k in range(N_s):
        k_vec = k_vecs[k, :] # shape: (2,)
        k_minus_kp_minus_G = k_vec[None, None, None, :] - k_vecs[None, None, :, :] - G_vecs[:, :, None, :] # shape: (N_grid, N_grid, N_s, 2)
        V_q = V(k_minus_kp_minus_G) # shape: (N_grid, N_grid, N_s)
        Lambda_k_kp_G_abs_square = np.abs(ac_ff[:, :, k, :]) ** 2 # shape: (N_grid, N_grid, N_s)
        fock_energies[k] = - (1 / A) * np.sum(
            V_q * Lambda_k_kp_G_abs_square,
        )
    assert np.all(np.abs(np.imag(fock_energies)) < 1e-6), f"Fock energies have significant imaginary parts: {np.max(np.abs(np.imag(fock_energies)))}"
    return np.real(fock_energies) # shape: (N_s,)

def hole_dispersion(
    bz: BrillouinZone2D,
    G_coords: np.ndarray,
    ac_ff: np.ndarray,
    V: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    G_vecs = bz.reciprocal_lattice.get_points(G_coords)
    assert G_coords.shape[0] == G_coords.shape[1]
    N_grid = G_coords.shape[0]
    G_radius = (N_grid - 1) // 2
    assert G_radius * 2 + 1 == N_grid
    assert np.all(np.abs(G_vecs[G_radius, G_radius]) < 1e-12)

    hartree_energies = hole_hartree_self_energies(
        bz, G_vecs, ac_ff, V
    )
    fock_energies = hole_fock_self_energies(
        bz, G_vecs, ac_ff, V
    )
    return -(hartree_energies + fock_energies)


if __name__ == "__main__":
    from functools import partial
    import time
    
    sqrt3 = np.sqrt(3)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    a_M = 1.0
    a1 = a_M * e2
    a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
    lattice = Lattice2D(np.stack([a1, a2]))
    b = (4 * np.pi) / (a_M * sqrt3)
    b1, b2 = lattice.reciprocal_lattice_vectors
    b3 = -(b1 + b2)

    # N = 27 grid
    t1 = (1 / 9) * (b1 - b2)
    t2 = (1 / 9) * (b1 - b3)

    sample_lattice_27 = Lattice2D(np.stack([t1, t2]))
    bz_27 = BrillouinZone2D(lattice, sample_lattice_27)
    N_s_27 = bz_27.n_samples

    # N = 28 grid
    p1 = b1 + t2 - t1
    normb1 = np.linalg.norm(b1)
    normp1 = np.linalg.norm(p1)
    distb1p1 = np.linalg.norm(b1 - p1)
    scale = normb1 / normp1
    rot = -np.arccos((normb1 ** 2 + normp1 ** 2 - distb1p1 ** 2) / (2 * normb1 * normp1))
    sample_lattice_28 = sample_lattice_27.transformed(scale=scale, rot=rot)

    bz_28 = BrillouinZone2D(lattice, sample_lattice_28)
    N_s_28 = bz_28.n_samples
    
    lB = 1.0
    # fourier_resolution = 128
    fourier_resolution = 1024
    K_func = partial(K_func1, args=(0.8, b1, b2, b3))
    
    # normalization constants test
    g_coords, wg = wg_fourier_components(
        K_func, bz_27.lattice, fourier_resolution, flatten=False
    )
    normalizations = acband_normalization_constants(
        bz_27, lB, g_coords, wg, eps=1e-5
    )
    
    # N = 27 test
    start = time.time()
    _, ac_ff_27 = acband_form_factors(
        bz_27,
        lB,
        K_func,
        fourier_resolution,
        G_radius=16,
        pbar=True
    )
    end = time.time()
    print(f"N = 27 AC band form factors computed in {end - start:.2f} seconds")
    print(f"{ac_ff_27.shape=}")
    
    # N = 28 test
    start = time.time()
    _, ac_ff_28 = acband_form_factors(
        bz_28,
        lB,
        K_func,
        fourier_resolution,
        G_radius=16,
        pbar=True
    )
    end = time.time()
    
    print(f"N = 28 AC band form factors computed in {end - start:.2f} seconds")
    print(f"{ac_ff_28.shape=}")

