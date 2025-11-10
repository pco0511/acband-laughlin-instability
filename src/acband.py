from typing import Callable

import numpy as np
from .qm_utils.lattice.lattice import Lattice2D
from .qm_utils.lattice.brillouin_zone import BrillouinZone2D

from einops import einsum, rearrange, reduce


def eta_factors(g_coords):
    m, n = rearrange(g_coords, "... mn -> mn ...")
    return (-1) ** (n + m + m * n)
    
def K_func1(x, args):
    K, g1, g2, g3 = args
    coeff = -np.sqrt(3) / (4 * np.pi) * K
    cos1 = np.cos(einsum("i d, d -> i", x, g1))
    cos2 = np.cos(einsum("i d, d -> i", x, g2))
    cos3 = np.cos(einsum("i d, d -> i", x, g3))
    return coeff * (cos1 + cos2 + cos3)

def K_func2(x, args):
    # solenoid
    pass


def fourier_components(
    K_func,
    lattice: Lattice2D, 
    resolution: int
):
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
    
    return g_coords, wg
    
def cross_z(arr1, arr2):
    return arr1[..., 0] * arr2[..., 1] - arr1[..., 1] * arr2[..., 0]

def normalization_constants(
    bz: BrillouinZone2D,
    lB: float,
    g_coords: np.ndarray, # shape: (res, res, 2)
    wg: np.ndarray # shape: (res, res)
):
    eta_vals = eta_factors(g_coords) # shape: (res, res)
    recip_lattice = bz.reciprocal_lattice
    gs = recip_lattice.get_points(g_coords)  # shape: (res, res, 2)
    ks = bz.k_points  # shape: (N_s, 2)
    k_cross_g = cross_z(ks[:, None, None, :], gs[None, ...])  # shape: (N_s, res, res)
    g_squared = reduce(gs ** 2, "m n d -> m n", "sum")  # shape: (res, res)
    
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
    g_coords: np.ndarray, # shape: (..., 2)
):
    pass
    
    

def acband_form_factors():
    pass


def projected_hamiltonian(
    bz: BrillouinZone2D,
    K_func: Callable[[np.ndarray], np.ndarray],
):
    pass