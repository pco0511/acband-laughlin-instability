from typing import Callable

import numpy as np
from .qm_utils.lattice.lattice import Lattice2D
from .qm_utils.lattice.brillouin_zone import BrillouinZone2D

from einops import einsum, rearrange, reduce


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
        args
    ) -> np.ndarray:
    # solenoid
    pass


def K_fourier_components(
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
    
    wg_raw = np.fft.fft2(K_vals) / (resolution ** 2)
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
    assert np.all(np.real(norm_invs2) > -eps), f"Normalization constants have non-positive real parts: {np.min(np.real(norm_invs2))}"
    
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
    res: np.ndarray, # 2^n - 2 is preferred
    eps: float = 1e-5
) -> np.ndarray:
    g_coords, wg = K_fourier_components(
        K_func, bz.lattice, res + 2, flatten=False
    ) # grid is extended by +-1 in each direction
    # shape: (res + 2, res + 2, 2), (res + 2, res + 2)
    normalizations = acband_normalization_constants(
        bz, lB, g_coords, wg, eps=eps
    ) # shape: (N_s,)
    ff_k_p_g = LLL_form_factors(
        bz, lB, g_coords
    ) # shape: (N_s, N_s, res + 2, res + 2)
    G_coords = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
    ])
    recip_lattice = bz.reciprocal_lattice
    
    Lambda_k_plus_G_p = np.zeros((G_coords.shape[0], bz.N_s, bz.N_s), dtype=np.complex128)# shape: (7, N_s, N_s)

    g_idx_left_center = 1
    g_idx_right_center = -2
    for iG, G in enumerate(G_coords):
        g_plus_G_slice_1 = slice(g_idx_left_center + G[0], g_idx_right_center + G[0] + 1)
        g_plus_G_slice_2 = slice(g_idx_left_center + G[1], g_idx_right_center + G[1] + 1)
        
        ff_k_p_g_plus_G = ff_k_p_g[:, :, g_plus_G_slice_1, g_plus_G_slice_2]  # shape: (N_s, N_s, res, res)
        N_k = normalizations[:, None]
        N_p = normalizations[None, :]
        unnormed = einsum(
            ff_k_p_g_plus_G,
            wg[g_plus_G_slice_1, g_plus_G_slice_2],
            "k p m n, m n -> k p"
        )
        Lambda_k_plus_G_p[iG, :, :] = N_k * N_p * unnormed
    
    return G_coords, Lambda_k_plus_G_p  

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
    # resolution = 126
    resolution = 1022
    K_func = partial(K_func1, args=(0.8, b1, b2, b3))
    
    # normalization constants test
    g_coords, wg = K_fourier_components(
        K_func, bz_27.lattice, resolution, flatten=False
    )
    normalizations = acband_normalization_constants(
        bz_27, lB, g_coords, wg, eps=1e-5
    )
    
    # N = 27 test
    start = time.time()
    ac_ff_27 = acband_form_factors(
        bz_27,
        lB,
        K_func,
        resolution
    )
    end = time.time()
    print(f"N = 27 AC band form factors computed in {end - start:.2f} seconds")
    print(f"{ac_ff_27.shape=}")
    
    # N = 28 test
    start = time.time()
    ac_ff_28 = acband_form_factors(
        bz_28,
        lB,
        K_func,
        resolution
    )
    end = time.time()
    
    print(f"N = 28 AC band form factors computed in {end - start:.2f} seconds")
    print(f"{ac_ff_28.shape=}")