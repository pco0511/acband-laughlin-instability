import itertools
from typing import Callable

import numpy as np
from .qm_utils.lattice.lattice import Lattice2D
from .qm_utils.lattice.brillouin_zone import BrillouinZone2D

from einops import einsum, rearrange, reduce


def eta_factors(
        g_coords: np.ndarray
    ) -> np.ndarray:
    m, n = rearrange(g_coords, "... d -> d ...")
    return (-1) ** (n + m + m * n)
    
def K_func1(
        x: np.ndarray,
        args
    ) -> np.ndarray:
    K, g1, g2, g3 = args
    coeff = -np.sqrt(3) / (4 * np.pi) * K
    cos1 = np.cos(einsum("i d, d -> i", x, g1))
    cos2 = np.cos(einsum("i d, d -> i", x, g2))
    cos3 = np.cos(einsum("i d, d -> i", x, g3))
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

def normalization_constants(
    bz: BrillouinZone2D,
    lB: float,
    g_coords: np.ndarray, # shape: (res, res, 2)
    wg: np.ndarray # shape: (res, res)
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
    
    assert np.all(np.abs(np.imag(norm_invs2)) < 1e-10), f"Normalization constants have significant imaginary parts"
    assert np.all(np.real(norm_invs2) > -1e-10), f"Normalization constants have non-positive real parts"
    
    return 1.0 / np.sqrt(np.abs(norm_invs2))

def LLL_form_factors(
    bz: BrillouinZone2D,
    lB: float,
    g_coords: np.ndarray, # shape: (N_g, 2)
) -> np.ndarray:
    assert g_coords.ndim == 2 and g_coords.shape[1] == 2, "g_coords must have shape (N_g, 2)"
    ks = bz.k_points  # shape: (N_s, 2)
    recip_lattice = bz.reciprocal_lattice
    gs = recip_lattice.get_points(g_coords) # (N_g, 2)

    lB2 = lB ** 2

    k1 = ks[:, None, :]  # shape: (N_s, 1, 2)
    k2 = ks[None, :, :]  # shape: (1, N_s, 2)
    k1_plus_k2 = k1 + k2 # shape: (N_s, N_s, 2)
    k1_minus_k2 = k1 - k2 # shape: (N_s, N_s, 2)

    exponent_imag_1 = (lB2 / 2) * cross_z(k1_plus_k2[:, :, None, :], gs[None, None, :, :])  # shape: (N_s, N_s, N_g)
    exponent_imag_2 = (lB2 / 2) * cross_z(k1, k2)[:, :, None] # shape: (N_s, N_s, 1)
    exponent_real = (-lB2 / 4) * norm_square(k1_minus_k2[:, :, None, :] - gs[None, None, :, :], axis=-1)  # shape: (N_s, N_s, N_g)
    
    eta_vals = eta_factors(g_coords)  # shape: (N_g,)

    ff_k1_k2_g = eta_vals[None, None, :] * np.exp(exponent_real + 1j * (exponent_imag_1 + exponent_imag_2))
    return ff_k1_k2_g  # shape: (N_s, N_s, N_g)

def acband_form_factors(
    bz: BrillouinZone2D,
    lB: float,
) -> np.ndarray:
    pass

def projected_hamiltonian(
    bz: BrillouinZone2D,
    K_func: Callable[[np.ndarray], np.ndarray],
):
    pass