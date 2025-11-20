import os
import pickle

import numpy as np
import netket as nk

from src.qm_utils.lattice.lattice import Lattice2D
from src.netket_compat import get_number_sector_indices, check_ordering
from brillouin_zones import construct_brillouin_zones


N_data_path = "data/supple_d/ed_spectrum_20_27_0.800_256_64_20251119_184801.pkl"
N_minus_2_data_path = "data/supple_d/ed_spectrum_18_27_0.800_256_64_20251119_202725.pkl"
N_s = 27

# N_data_path = "data/supple_d/ed_spectrum_21_28_0.800_256_64_20251119_191738.pkl"
# N_minus_2_data_path = "data/supple_d/ed_spectrum_19_28_0.800_256_64_20251120_004155.pkl"
# N_s = 28

# load pickle
with open(N_data_path, 'rb') as f:
    N_data = pickle.load(f)

with open(N_minus_2_data_path, 'rb') as f:
    N_minus_2_data = pickle.load(f)

"""
data format:
{
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
}
"""

if "bz" in N_data:
    bz = N_data['bz']
else:
    a_M = 1
    sqrt3 = np.sqrt(3)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    a1 = a_M * e2
    a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
    lattice = Lattice2D(np.stack([a1, a2]))
    bz = construct_brillouin_zones(lattice)
    bz = bz[N_s]

# get ground state vector
GS_vec_N = N_data['eigenvectors'][0][:, 0]  # ground state vector for N particles
GS_vec_N_minus_2 = N_minus_2_data['eigenvectors'][0][:, 0]  # ground state vector for N-2 particles


# construct sparse representation of pairing gap function
full_fock = nk.hilbert.SpinOrbitalFermions(N_s, s=None, n_fermions=None)
N_subspaces = nk.hilbert.SpinOrbitalFermions(N_s, s=None, n_fermions=N_s)
N_minus_2_subspaces = nk.hilbert.SpinOrbitalFermions(N_s, s=None, n_fermions=N_s - 2)

from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag

pair_ann_sparse = []

for k in range(N_s):
    neg_k = bz.neg(k)
    op = c(full_fock, neg_k) @ c(full_fock, k)
    full_sparse = op.to_sparse()
    # pair_ann_sparse.append()



# calculate pairing gap function




# visualize pairing gap function