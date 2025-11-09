

import netket as nk


hilb = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=27, s=0, n_fermions=18
)

for state in hilb.states():
    print(state)