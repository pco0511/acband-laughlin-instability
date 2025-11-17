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
from src.qm_utils.lattice.brillouin_zone import BrillouinZone2D
from src.netket_compat import get_sector_constraints
from src.acband import acband_form_factors, K_func1, K_func2



parser = argparse.ArgumentParser(description="ED parameters")
parser.add_argument("--n_sites", "-N", type=int, choices=[25, 27, 28], default=27, help="Number of lattice sites (N_s); must be one of {25, 27, 28}")
parser.add_argument("--n_fermions", "-F", type=int, default=18, help="Number of fermions (N_f)")
parser.add_argument("--K", type=float, default=0.8, help="K parameter for K_func")
args = parser.parse_args()

N_s = args.n_sites
N_f = args.n_fermions
K = args.K

sqrt3 = 3.0 ** 0.5

lB = 1.0
a_M = (((4 * np.pi) / sqrt3) ** 0.5) * lB
# a_M = 1
# lB = ((sqrt3 / (4 * np.pi)) ** 0.5) * a_M


# fourier_resolution = 128
fourier_resolution = 256
G_radius = 64
V1 = 1.0
v1 = 3 * V1 * (a_M ** 4) / (4 * np.pi)

bz: dict[int, BrillouinZone2D] = {}

e1 = np.array([1, 0])
e2 = np.array([0, 1])
a1 = a_M * e2
a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
lattice = Lattice2D(np.stack([a1, a2]))
recip_lattice = lattice.reciprocal()
b = (4 * np.pi) / (a_M * sqrt3)
b1, b2 = lattice.reciprocal_lattice_vectors
b3 = -(b1 + b2)

# N = 27 grid
t1 = (1 / 9) * (b1 - b2)
t2 = (1 / 9) * (b1 - b3)
t3 = (1 / 9) * (b2 - b3)
sample_lattice_27 = Lattice2D(np.stack([t1, t2]))
bz_27 = BrillouinZone2D(lattice, sample_lattice_27)
bz[27] = bz_27

# N = 28 grid
p1 = b1 + t2 - t1
normb1 = np.linalg.norm(b1)
normp1 = np.linalg.norm(p1)
distb1p1 = np.linalg.norm(b1 - p1)
scale = normb1 / normp1
rot = -np.arccos((normb1 ** 2 + normp1 ** 2 - distb1p1 ** 2) / (2 * normb1 * normp1))
sample_lattice_28 = sample_lattice_27.transformed(scale=scale, rot=rot)

bz_28 = BrillouinZone2D(lattice, sample_lattice_28)
bz[28] = bz_28

# N = 25 grid
p1 = b1 + 2 * t3 - t1
normb1 = np.linalg.norm(b1)
normp1 = np.linalg.norm(p1)
distb1p1 = np.linalg.norm(b1 - p1)
scale = normb1 / normp1
rot = -np.arccos((normb1 ** 2 + normp1 ** 2 - distb1p1 ** 2) / (2 * normb1 * normp1))
sample_lattice_25 = sample_lattice_27.transformed(scale=scale, rot=rot)

bz_25 = BrillouinZone2D(lattice, sample_lattice_25)
bz[25] = bz_25

bz_N_s = bz[N_s]

constraints = get_sector_constraints(bz_N_s, N_f)
hilbs = [
    nk.hilbert.SpinOrbitalFermions(
        n_orbitals=N_s, n_fermions=N_f, constraint=constraint
    ) for constraint in constraints
]

for k_index, sector in enumerate(hilbs):
    print(f"Sector {k_index}:")
    print(f"  Total Momentum: {bz_N_s[k_index]}")
    print(f"  Dimension: {sector.n_states}")

A = N_s * lattice.unit_cell_area

def V(q):
    return -v1 * np.linalg.norm(q, axis=-1) ** 2
 
    
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

G_vecs = recip_lattice.get_points(G_coords)
start_idx = 1
end_idx = 2 * G_radius
G_vecs_slice = G_vecs[start_idx:end_idx, start_idx:end_idx]

print(f"N_s = {N_s} AC band form factors computed in {end - start:.2f} seconds")
print(f"{ac_ff.shape=}")

hamiltonians = []
for sector_index, sector in enumerate(hilbs):
    print(f"Constructing Hamiltonian for sector {sector_index}...")
    H = 0.0
    for k, p, q in itertools.product(range(N_s), repeat=3):
        k1 = bz_N_s.sum(k, q) # k + q
        k2 = bz_N_s.sub(p, q) # p - q
        k3 = p
        k4 = k
        
        k1_coord = bz_N_s.k_coords[k1]
        k2_coord = bz_N_s.k_coords[k2]
        k3_coord = bz_N_s.k_coords[k3]
        k4_coord = bz_N_s.k_coords[k4]
        q_coord = bz_N_s.k_coords[q]
        q_vec = bz_N_s.k_points[q]

        _, G1_coord = bz_N_s.fold_coord(k4_coord + q_coord) # k1_coord + G1 = k4 + q
        _, G2_coord = bz_N_s.fold_coord(k3_coord - q_coord) # k2_coord + G2 = k3 - q

        q_vec_shifted_grid = G_vecs_slice + q_vec
        V_grid = V(q_vec_shifted_grid)
        
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

        interaction_grid = V_grid * Lambda_1_grid * Lambda_2_grid
        coeff = (1 / (2 * A)) * np.sum(interaction_grid)

        c_dag_k1 = cdag(sector, k1)
        c_dag_k2 = cdag(sector, k2)
        c_k3 = c(sector, k3)
        c_k4 = c(sector, k4)

        H += complex(coeff) * (c_dag_k1 @ c_dag_k2 @ c_k3 @ c_k4)
    H = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(H)
    hamiltonians.append(H)

eigenvalues = []
eigenvectors = []
com_momentums = []

for k_index, (sector, H) in enumerate(zip(hilbs, hamiltonians)):
    print(f"Diagonalizing sector {k_index} with dimension {sector.n_states}...")
    start = time.time()
    evals, evecs = nk.exact.lanczos_ed(H, k=10, compute_eigenvectors=True)
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
plt.scatter(k_coms_flatten, energies_flatten, color='red', alpha=0.5)
plt.xlabel(r'$|\mathbf{k}_{\mathrm{COM}}|$')
plt.ylabel('Energy')
plt.title(f'ED Spectrum ($N={N_f}$, $N_S={N_s}$)')
os.makedirs('figs/main', exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
plt.savefig(f'figs/main/ed_spectrum_{N_f}_{N_s}_{K:.3f}_{ts}.png', dpi=300)
plt.close()

data_path = f'data/main/ed_spectrum_{N_f}_{N_s}_{K:.3f}_{ts}.pkl'
os.makedirs('data/main', exist_ok=True)
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
    }, f)