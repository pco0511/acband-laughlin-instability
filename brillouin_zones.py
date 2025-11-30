
import os
import numpy as np

from src.qm_utils.lattice.lattice import Lattice2D
from src.qm_utils.lattice.brillouin_zone import BrillouinZone2D


def construct_brillouin_zones(lattice: Lattice2D) -> dict[int, BrillouinZone2D]:
    bz: dict[int, BrillouinZone2D] = {}

    b1, b2 = lattice.reciprocal_lattice_vectors
    b3 = -(b1 + b2)

    t1 = (1 / 9) * (b1 - b2)
    t2 = (1 / 9) * (b1 - b3)
    t3 = (1 / 9) * (b2 - b3)

    # N = 27 grid
    sample_lattice_27 = Lattice2D(np.stack([t1, t2]))
    bz_27 = BrillouinZone2D(lattice, sample_lattice_27)
    bz[27] = bz_27

    def construct_scaled_rotated_bz(p1: np.ndarray) -> BrillouinZone2D:
        normb1 = np.linalg.norm(b1)
        normp1 = np.linalg.norm(p1)
        distb1p1 = np.linalg.norm(b1 - p1)
        scale = normb1 / normp1
        rot = -np.arccos((normb1 ** 2 + normp1 ** 2 - distb1p1 ** 2) / (2 * normb1 * normp1))
        sample_lattice_new = sample_lattice_27.transformed(scale=scale, rot=rot)
        return BrillouinZone2D(lattice, sample_lattice_new)
    # N = 28 grid
    p1 = b1 + t3
    bz[28] = construct_scaled_rotated_bz(p1)

    # N = 25 grid
    p1 = 5 * t1
    bz[25] = construct_scaled_rotated_bz(p1)

    # N = 36 grid
    p1 = 6 * t1
    bz[36] = construct_scaled_rotated_bz(p1)

    # N = 39 grid
    p1 = b1 + t2 + t3
    bz[39] = construct_scaled_rotated_bz(p1)

    # N = 48 grid
    sample_lattice_48 = sample_lattice_27.transformed(scale=3/4)
    bz_48 = BrillouinZone2D(lattice, sample_lattice_48)
    bz[48] = bz_48

    # N = 49 grid
    p1 = 7 * t1
    bz[49] = construct_scaled_rotated_bz(p1)
    
    # N = 52 grid
    p1 = 6 * t1 + 2 * t2
    bz[52] = construct_scaled_rotated_bz(p1)

    # N = 57 grid
    p1 = 7 * t1 + t2
    bz[57] = construct_scaled_rotated_bz(p1)

    # N = 63 grid
    p1 = 3 * t1 + 6 * t2
    bz[63] = construct_scaled_rotated_bz(p1)

    # N = 64 grid
    p1 = 8 * t2
    bz[64] = construct_scaled_rotated_bz(p1)

    return bz

def main(arg):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from src.qm_utils.lattice.lattice import Lattice2D
    
    img_root_dir, = arg

    sqrt3 = np.sqrt(3)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    a_M = 1.0
    a1 = a_M * e2
    a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
    lattice = Lattice2D(np.stack([a1, a2]))
    b = (4 * np.pi) / (a_M * sqrt3)

    bz = construct_brillouin_zones(lattice)

    R = b / sqrt3
    thetas = np.linspace(np.pi / 2, 5 * np.pi / 2, 7)
    hexagon_x = R * np.cos(thetas)
    hexagon_y = R * np.sin(thetas)

    for n, bz_n in bz.items():
        k_points = bz_n.k_points
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        ax.set_title(f"Brillouin Zone $N_s={n}$", fontsize=16)
        ax.set_aspect('equal')
        k_x = k_points[:, 0]
        k_y = k_points[:, 1]
        ax.plot(hexagon_x, hexagon_y, color='k')
        ax.scatter(k_x, k_y, s=250, c='k')
        for i in range(n):
            x = k_x[i]
            y = k_y[i]
            ax.text(x, y, f"{i}", fontsize=8, color="w", ha='center', va='center')
        ax.set_xlim(-36 / b, 36 / b)
        ax.set_ylim(-36 / b, 36 / b)
        fig_path = os.path.join(img_root_dir, f"brillouin_zone_Ns_{n}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root_dir", default="figs/bz", type=str)
    args = parser.parse_args()
    main([args.img_root_dir])