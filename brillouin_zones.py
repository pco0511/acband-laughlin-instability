
import numpy as np

from src.qm_utils.lattice.lattice import Lattice2D
from src.qm_utils.lattice.brillouin_zone import BrillouinZone2D

def construct_brillouin_zones(lattice: Lattice2D) -> dict[int, BrillouinZone2D]:
    bz: dict[int, BrillouinZone2D] = {}

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

    return bz