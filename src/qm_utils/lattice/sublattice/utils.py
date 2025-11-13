from numpy import np

from ..lattice import Lattice2D

def check_sublattice(
    lattice: Lattice2D,
    sublattice: Lattice2D,
    eps: float=1e-12
) -> bool:
    L1, L2 = sublattice.lattice_vectors

    (m11, m12), offset1 = lattice.divmod(L1)
    (m21, m22), offset2 = lattice.divmod(L2)

    if np.linalg.norm(offset1) >= eps or np.linalg.norm(offset2) >= eps:
        return False
    return True

def get_coeff_matrix(
    lattice: Lattice2D,
    sublattice: Lattice2D,
    ensure_sublattice: bool=False,
    eps: float=1e-12
) -> np.ndarray:
    L1, L2 = sublattice.lattice_vectors

    (m11, m12), offset1 = lattice.divmod(L1)
    (m21, m22), offset2 = lattice.divmod(L2)

    if not ensure_sublattice:
        if np.linalg.norm(offset1) > eps or np.linalg.norm(offset2) > eps:
            raise ValueError("Sublattice is not a sublattice of given lattice.")

    return np.array([[m11, m12], [m21, m22]])


def get_points_in_supercell(
    lattice: Lattice2D,
    sublattice: Lattice2D,
    ensure_sublattice: bool=False,
    eps: float=1e-12
) -> np.ndarray:
    if not ensure_sublattice:
        check_sublattice(
            lattice,
            sublattice,
            eps=eps
        )
    
    a1, a2 = lattice.lattice_vectors
    L1, L2 = sublattice.lattice_vectors

    A = max(np.linalg.norm(a1), np.linalg.norm(a2))
    L = min(np.linalg.norm(L1), np.linalg.norm(L2))
    M = np.ceil(L / A).astype(int)

    xx = np.linspace(-M, M, 2 * M + 1)
    yy = np.linspace(-M, M, 2 * M + 1)
    mgrid = np.meshgrid(xx, yy)
    candidates = lattice.get_points(*mgrid, flatten=False)

    candidates = np.reshape(candidates, (-1, 2))

    coords, _ = sublattice.divmod(candidates)
    wigner_seitz_mask = np.logical_and(coords[:, 0]==0, coords[:, 1]==0)

    return coords[wigner_seitz_mask], candidates[wigner_seitz_mask]
