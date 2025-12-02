import argparse
from datetime import datetime
from functools import partial
import itertools
import time
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import jax
jax.config.update("jax_enable_x64", True)
import logging
logger = logging.getLogger("jax._src.xla_bridge")
logger.setLevel(logging.ERROR)

import netket as nk
from netket.hilbert import SpinOrbitalFermions
from netket.operator import FermionOperator2nd
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd

from src.qm_utils.lattice.lattice import Lattice2D
from src.netket_compat import csr_from_nk_fermion_op
from src.acband import (
    acband_form_factors, 
    interaction_matrix, 
    K_func1,
    K_func2,
    interaction_hamiltonian_terms
)
from src.qm_utils.fermion.fermionic_fock import DiscreteFermionicFockSpace
from src.qm_utils.fermion.fermion_utils import (
    total_sum_by_table

)
from brillouin_zones import construct_brillouin_zones


total_start = time.time()

parser = argparse.ArgumentParser(description="ED parameters")
parser.add_argument("--save_name", type=str, default="main", help="Base name for saving figures and data")
parser.add_argument("--n_sites", type=int, choices=[25, 27, 28], default=27, help="Number of lattice sites (N_s); must be one of {25, 27, 28}")
parser.add_argument("--n_fermions", type=int, default=9, help="Number of fermions (N_f)")
parser.add_argument("--sum_radius", type=float, default=10.0, help="Sum radius for attached solenoids")
parser.add_argument("--sigma", type=float, required=True, help="Sigma for attached solenoids")
parser.add_argument("--fourier_resolution", type=int, default=128, help="Fourier resolution for AC band form factors")
parser.add_argument("--G_radius", type=int, default=32, help="G vector radius for AC band form factors")
parser.add_argument("--num_evecs", type=int, default=15, help="Number of eigenvalues/eigenvectors to compute per sector")
parser.add_argument("--gamma_only", action="store_true", help="If set, only diagonalize the Gamma sector")
parser.add_argument("--simple_out", action="store_true", help="If set, produce simple output")
parser.add_argument("--quiet", action="store_true", help="If set, suppress all output except for errors")
args = parser.parse_args()

simple_out = args.simple_out
quiet = args.quiet

N_s = args.n_sites
N_f = args.n_fermions
m = round(N_s / N_f)
if N_s % N_f != 0:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"WARNING: N_s must be divisible by N_f for 1/m filling; got N_s={N_s}, N_f={N_f}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    

sqrt3 = 3.0 ** 0.5

# lB = 1.0
# a_M = (((4 * np.pi) / sqrt3) ** 0.5) * lB
a_M = 1
lB = ((sqrt3 / (4 * m * np.pi)) ** 0.5) * a_M
     
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
if N_s not in bz:
    raise ValueError(f"Brillouin zone for N_s={N_s} not found. Available: {list(bz.keys())}")
bz_N_s = bz[N_s]

# Many-body Hilbert spaces
full_hilb = DiscreteFermionicFockSpace(mode_labels=list(range(N_s)), particle_numbers=N_f)

if not quiet and not simple_out:
    print("Labeling states...")
start = time.time()
state_labels = total_sum_by_table(
    full_hilb.states,
    n_modes=N_s, zero_idx=bz_N_s.zero(), sum_table=bz_N_s.sum_table
)
end = time.time()
if not quiet and not simple_out:
    print(f"  Labeled {full_hilb.dim} states in {end - start:.2f} seconds")

if not quiet and not simple_out:
    print("Decomposing sectors...")
start = time.time()
sector_labels = list(range(N_s))
sectors = full_hilb.decompose_sector_by_labels(state_labels, sector_labels)
end = time.time()
if not quiet and not simple_out:
    print(f"  Decomposed into {len(sectors)} sectors in {end - start:.2f} seconds")
    for label, sector in sectors.items():
        print(f"    Sector {label}: dimension {sector.dim}")

 
b1, b2 = lattice.reciprocal_lattice_vectors
b3 = -(b1 + b2)

sigma = args.sigma
sum_radius = args.sum_radius
K_func_args = (3, sigma, sum_radius, a1, a2)
K_func = partial(K_func2, args=K_func_args)
# zero_K_func = partial(K_func1, args=(0, b1, b2, b3))

if not quiet and not simple_out:
    print("computing AC band form factors...")
start = time.time()
G_coords, ac_ff = acband_form_factors(
    bz[N_s],
    lB,
    K_func,
    fourier_resolution,
    G_radius=G_radius
)
end = time.time()
if not quiet and not simple_out:
    print(f"AC band form factors computed in {end - start:.2f} seconds")

G_vecs = recip_lattice.get_points(G_coords)
start_idx = 1
end_idx = 2 * G_radius
G_vecs_slice = G_vecs[start_idx:end_idx, start_idx:end_idx]

def V(q):
    return -v1 * np.linalg.norm(q, axis=-1) ** 2

if not quiet and not simple_out:
    print("computing interaction matrix...")
start = time.time()
int_mat = interaction_matrix(
    bz_N_s,
    G_coords,
    ac_ff,
    V
)
end = time.time()
if not quiet and not simple_out:
    print(f"Interaction matrix computed in {end - start:.2f} seconds")

nk_kilb = SpinOrbitalFermions(
    n_orbitals=N_s, s=None, n_fermions=N_f
)
terms, weights = interaction_hamiltonian_terms(bz_N_s, int_mat)
H = FermionOperator2nd(
    nk_kilb,
    terms,
    weights
)
H = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(H)

eigenvalues = []
eigenvectors = []
com_momentums = []

if not quiet and simple_out:
    elapsed = time.time() - total_start
    print(f"The pre-calculation for the diagonalization is done in {elapsed:.2f} seconds.\n Starting diagonalization...")

gamma_idx = bz_N_s.zero()
for k_idx, (label, sector) in enumerate(sectors.items()):
    if args.gamma_only and label != gamma_idx:
        continue
    if not quiet and not simple_out:
        print(f"Diagonalizing sector {label} with dimension {sector.dim}...")
        print("  Constructing sparse representation:")
    start = time.time()
    sparse_rep = csr_from_nk_fermion_op(
        H, sector, sector, batch_size=512, pbar=True, n_jobs=32
    )
    end = time.time()
    if not quiet and not simple_out:
        print(f"  Sparse matrix constructed in {end - start:.2f} seconds")

    if not quiet and not simple_out:
        print("  Diagonalizing...")
    start = time.time()
    evals, evecs = scipy.sparse.linalg.eigsh(sparse_rep, k=args.num_evecs, which='SA', return_eigenvectors=True)
    end = time.time()
    if not quiet:
        if simple_out:
            print(f"  Diagonalized in {end - start:.2f} seconds: sector {label} with dimension {sector.dim}")
        else:
            print(f"  Diagonalized in {end - start:.2f} seconds")
    
    eigenvalues.append(evals)
    eigenvectors.append(evecs)
    com_momentums.append(bz_N_s.k_points[k_idx])
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
plt.title(f'ED Spectrum ($N={N_f}$, $N_S={N_s}$, $\\sigma={sigma:.2f}$)')


save_name = args.save_name
FIG_ROOT = f'figs/{save_name}'
DATA_ROOT = f'data/{save_name}'

file_name = f'ed_spectrum_{N_f}_{N_s}_{sigma:.3f}_{fourier_resolution}_{G_radius}'

os.makedirs(FIG_ROOT, exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
fig_path = os.path.join(FIG_ROOT, f'{file_name}_{ts}.png')
if not quiet and not simple_out:
    print(f"Saving figure to {fig_path}...")
plt.savefig(fig_path, dpi=300, transparent=True)
plt.close()

data_path = os.path.join(DATA_ROOT, f'{file_name}_{ts}.pkl')
if not quiet and not simple_out:
    print(f"Saving data to {data_path}...")
os.makedirs(DATA_ROOT, exist_ok=True)
with open(data_path, 'wb') as f:
    pickle.dump({
        'k_coms': k_coms_flatten,
        'energies': energies_flatten,
        'N_f': N_f,
        'N_s': N_s,
        'm': m,
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

total_end = time.time()
total_time = total_end - total_start
if not quiet:
    print(f"Calculation completed in {total_time:.2f} seconds")
