import argparse
from datetime import datetime
from functools import partial
import itertools
import time
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import netket as nk
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd

from src.qm_utils.lattice.lattice import Lattice2D
from src.netket_compat import get_sector_constraints
from src.acband import acband_form_factors, interaction_matrix, K_func1, K_func2

from brillouin_zones import construct_brillouin_zones

parser = argparse.ArgumentParser(description="ED parameters")
parser.add_argument("--save_name", type=str, default="main", help="Base name for saving figures and data")
parser.add_argument("--n_sites", "-N", type=int, choices=[25, 27, 28], default=27, help="Number of lattice sites (N_s); must be one of {25, 27, 28}")
parser.add_argument("--n_fermions", "-F", type=int, default=18, help="Number of fermions (N_f)")
parser.add_argument("--K", type=float, default=0.8, help="K parameter for K_func")
parser.add_argument("--fourier_resolution", type=int, default=256, help="Fourier resolution for AC band form factors")
parser.add_argument("--G_radius", type=int, default=64, help="G vector radius for AC band form factors")
args = parser.parse_args()

N_s = args.n_sites
N_f = args.n_fermions
K = args.K

sqrt3 = 3.0 ** 0.5

# lB = 1.0
# a_M = (((4 * np.pi) / sqrt3) ** 0.5) * lB
a_M = 1
lB = ((sqrt3 / (4 * np.pi)) ** 0.5) * a_M
     
# fourier_resolution = 128
fourier_resolution = args.fourier_resolution
G_radius = args.G_radius
V1 = 1.0
# v1 = 3 * V1 * (a_M ** 4) / (4 * np.pi)
v1 = 3 * V1 * (a_M ** 4) / (8 * np.pi) # ????? 4 pi -> 8 pi


# Lattice and Brillouin zones
e1 = np.array([1, 0])
e2 = np.array([0, 1])
a1 = a_M * e2
a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
lattice = Lattice2D(np.stack([a1, a2]))
recip_lattice = lattice.reciprocal()

bz = construct_brillouin_zones(lattice)

bz_N_s = bz[N_s]


# Many-body Hilbert spaces
constraints = get_sector_constraints(bz_N_s, N_f)
hilbs = [
    nk.hilbert.SpinOrbitalFermions(
        n_orbitals=N_s, n_fermions=N_f, constraint=constraint
    ) for constraint in constraints
]

for k_index, sector in enumerate(hilbs):
    print(f"Constructing Sector {k_index}:")
    print(f"  Total Momentum: {bz_N_s[k_index]}")
    print(f"  Dimension: {sector.n_states}")
 
b1, b2 = lattice.reciprocal_lattice_vectors
b3 = -(b1 + b2)

K_func_args = (K, b1, b2, b3)
K_func = partial(K_func1, args=K_func_args)

# K_func_args = (3, 0.25, 20, a1, a2)
# K_func = partial(K_func2, args=K_func_args)

start = time.time()
G_coords, ac_ff = acband_form_factors(
    bz[N_s],
    lB,
    K_func,
    fourier_resolution,
    G_radius=G_radius,
    pbar=True
)
end = time.time()
print(f"AC band form factors computed in {end - start:.2f} seconds")

G_vecs = recip_lattice.get_points(G_coords)
start_idx = 1
end_idx = 2 * G_radius
G_vecs_slice = G_vecs[start_idx:end_idx, start_idx:end_idx]

# Interaction matrix
# def V(q):
#     return -v1 * np.linalg.norm(q, axis=-1) ** 2

def V(q):
    return -v1 * np.linalg.norm(q, axis=-1) ** 2

    

start = time.time()
int_mat = interaction_matrix(
    bz_N_s,
    G_coords,
    ac_ff,
    V
)
end = time.time()
print(f"Interaction matrix computed in {end - start:.2f} seconds")

hamiltonians = []
for sector_index, sector in enumerate(hilbs):
    H = 0.0
    for k, p, q in itertools.product(range(N_s), repeat=3):
        H_k1_k2_k3_k4 = int_mat[k, p, q]
        k1 = bz_N_s.sum(k, q) # k + q
        k2 = bz_N_s.sub(p, q) # p - q
        k3 = p
        k4 = k
        c_dag_k1 = cdag(sector, k1)
        c_dag_k2 = cdag(sector, k2)
        c_k3 = c(sector, k3)
        c_k4 = c(sector, k4)
        H += complex(H_k1_k2_k3_k4) * (c_dag_k1 @ c_dag_k2 @ c_k3 @ c_k4)
    H = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(H)
    hamiltonians.append(H)

eigenvalues = []
eigenvectors = []
com_momentums = []

for k_index, (sector, H) in enumerate(zip(hilbs, hamiltonians)):
    print(f"Diagonalizing sector {k_index} with dimension {sector.n_states}...")
    start = time.time()
    evals, evecs = nk.exact.lanczos_ed(H, k=30, compute_eigenvectors=True)
    end = time.time()
    print(f"  Diagonalized in {end - start:.2f} seconds")
    eigenvalues.append(evals)
    eigenvectors.append(evecs)
    com_momentums.append(bz_N_s.k_points[k_index])

k_coms_flatten = []
energies_flatten = []

for com_momentum, spectrum in zip(com_momentums, eigenvalues):
    k_coms_flatten.extend([np.linalg.norm(com_momentum)] * len(spectrum))
    energies_flatten.extend(spectrum.tolist())
    
k_coms_flatten = np.array(k_coms_flatten)
energies_flatten = np.array(energies_flatten)
energies_flatten -= np.min(energies_flatten)

plt.figure(figsize=(8, 6))
plt.scatter(k_coms_flatten * lB, energies_flatten, color='red', alpha=0.25)
plt.xlabel(r'$|\mathbf{k}_{\mathrm{COM}}|$')
plt.ylabel('Energy')
plt.title(f'ED Spectrum ($N={N_f}$, $N_S={N_s}$, $K={K:.3f}$)')


FIG_ROOT = 'figs/supple_d'
DATA_ROOT = 'data/supple_d'

file_name = f'ed_spectrum_{N_f}_{N_s}_{K:.3f}_{fourier_resolution}_{G_radius}'

os.makedirs(FIG_ROOT, exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
fig_path = os.path.join(FIG_ROOT, f'{file_name}_{ts}.png')
plt.savefig(fig_path, dpi=300, transparent=True)
plt.close()

data_path = os.path.join(DATA_ROOT, f'{file_name}_{ts}.pkl')
os.makedirs(DATA_ROOT, exist_ok=True)
with open(data_path, 'wb') as f:
    pickle.dump({
        'k_coms': k_coms_flatten,
        'energies': energies_flatten,
        'N_f': N_f,
        'N_s': N_s,
        'K': K,
        'fourier_resolution': fourier_resolution,
        'G_radius': G_radius,
        'a_M': a_M,
        'lB': lB,
        'com_momentums': com_momentums,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'lattice': lattice,
        'recip_lattice': recip_lattice,
        'bz': bz_N_s,
    }, f)