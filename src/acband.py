import itertools
from typing import Callable

import numpy as np

from tqdm.auto import tqdm, trange

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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

def LLL_form_factors(
    bz: BrillouinZone2D,
    lB: float,
    g_coords: np.ndarray, # shape: (..., 2)
) -> np.ndarray:
    g_batch_shape = g_coords.shape[:-1]
    g_coords_flatten = rearrange(g_coords, "... d -> (...) d")
    
    ks = bz.k_points  # shape: (N_s, 2)
    recip_lattice = bz.reciprocal_lattice
    gs = recip_lattice.get_points(g_coords_flatten) # (N_g, 2)
    lB2 = lB ** 2

    k1 = ks[:, None, :]  # shape: (N_s, 1, 2)
    k2 = ks[None, :, :]  # shape: (1, N_s, 2)
    k1_plus_k2 = k1 + k2 # shape: (N_s, N_s, 2)
    k1_minus_k2 = k1 - k2 # shape: (N_s, N_s, 2)

    exponent_imag_1 = (lB2 / 2) * cross_z(k1_plus_k2[:, :, None, :], gs[None, None, :, :])  # shape: (N_s, N_s, N_g)
    exponent_imag_2 = (lB2 / 2) * cross_z(k1, k2)[:, :, None] # shape: (N_s, N_s, 1)
    exponent_real = (-lB2 / 4) * norm_square(k1_minus_k2[:, :, None, :] - gs[None, None, :, :], axis=-1)  # shape: (N_s, N_s, N_g)
    
    eta_vals = eta_factors(g_coords_flatten)  # shape: (N_g,)

    ff_k1_k2_g_flatten = eta_vals[None, None, :] * np.exp(exponent_real + 1j * (exponent_imag_1 + exponent_imag_2)) # shape: (N_s, N_s, N_g)
    ff_k1_k2_g = ff_k1_k2_g_flatten.reshape(bz.N_s, bz.N_s, *g_batch_shape)  # shape: (N_s, N_s, ...)
    return ff_k1_k2_g

@njit(parallel=True, fastmath=True, cache=True)
def _compute_lambda_kernel(
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


def acband_form_factors(
    bz: BrillouinZone2D,
    lB: float,
    K_func: Callable[[np.ndarray], np.ndarray],
    fourier_resolution: int, # 2^n is preferred
    G_radius: int,
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

    Lambda_k_p_G = _compute_lambda_kernel(
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

    assert np.all(np.abs(np.imag(hartree_energies)) < 1e-10), "Hartree energies have significant imaginary parts"

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
    assert np.all(np.abs(np.imag(fock_energies)) < 1e-10), "Fock energies have significant imaginary parts"
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

