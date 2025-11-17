import itertools
from typing import Callable

import numpy as np

from .qm_utils.lattice.lattice import Lattice2D
from .qm_utils.lattice.brillouin_zone import BrillouinZone2D

from einops import einsum, rearrange, pack


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
    coeff = -np.sqrt(3) / (4 * np.pi) * K
    cos1 = np.cos(einsum(x, b1, "... d, d -> ..."))
    cos2 = np.cos(einsum(x, b2, "... d, d -> ..."))
    cos3 = np.cos(einsum(x, b3, "... d, d -> ..."))
    return coeff * (cos1 + cos2 + cos3)

def K_func2(
        x: np.ndarray,
        args: tuple[int, float, int, np.ndarray, np.ndarray]
    ) -> np.ndarray:
    # solenoid
    m, sigma, N, a1, a2 = args
    ns = np.linspace(-N, N, 2 * N + 1)
    n1 = ns[:, None, None]
    n2 = ns[None, :, None]
    lattice_points = n1 * a1[None, None, :] + n2 * a2[None, None, :]
    flatten_lattice_points = rearrange(lattice_points, "n1 n2 d -> (n1 n2) d")
    
    x_batch_shape = x.shape[:-1]
    x_flatten = rearrange(x, "... d -> (...) d")
    diffs = x_flatten[:, None, :] - flatten_lattice_points[None, :, :]
    dists_square = np.sum(diffs ** 2, axis=-1) / (sigma ** 2)
    dists = np.sqrt(dists_square) 

    chis = np.where(dists < 1, dists_square - (1/2), np.log(dists))
    k_vals_flatten = (1 / m) * np.sum(chis, axis=1)
    k_vals = np.reshape(k_vals_flatten, shape=x_batch_shape)
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
    
    assert np.all(np.abs(np.imag(norm_invs2)) < eps), f"Normalization constants have significant imaginary parts"
    assert np.all(np.real(norm_invs2) > eps), f"Normalization constants have non-positive real parts: {np.min(np.real(norm_invs2))}"
    
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

def acband_form_factors(
    bz: BrillouinZone2D,
    lB: float,
    K_func: Callable[[np.ndarray], np.ndarray],
    fourier_resolution: int, # 2^n is preferred
    G_radius: int,
    pbar: bool=False,
    eps: float=1e-10
) -> np.ndarray:
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
    
    Lambda_k_p_G = np.zeros((N_grid, N_grid, bz.N_s, bz.N_s), dtype=np.complex128)

    if pbar:
        from tqdm import tqdm
        G_indices = tqdm(list(itertools.product(range(N_grid), repeat=2)), desc="Computing AC band form factors")
    else:
        G_indices = itertools.product(range(N_grid), repeat=2)
        
    for i, j in G_indices:
        m, n = G_coords[i, j]
        start1 = G_radius + m
        stop1 = fourier_resolution - G_radius + m
        g_plus_G_slice_1 = slice(start1, stop1)
        
        start2 = G_radius + n
        stop2 = fourier_resolution - G_radius + n
        g_plus_G_slice_2 = slice(start2, stop2)
        
        ff_k_p_g_plus_G = ff_k_p_g[:, :, g_plus_G_slice_1, g_plus_G_slice_2]
        N_k = normalizations[:, None]
        N_p = normalizations[None, :]
        unnormed = einsum(
            ff_k_p_g_plus_G,
            wg[G_radius:-G_radius, G_radius:-G_radius],
            "k p m n, m n -> k p"
        )
        Lambda_k_p_G[i, j, :, :] = N_k * N_p * unnormed
    
    return G_coords, Lambda_k_p_G

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