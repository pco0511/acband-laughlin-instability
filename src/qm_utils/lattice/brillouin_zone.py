import itertools
import random

import numpy as np
from einops import rearrange, pack

if __name__ == "__main__":
    from lattice import Lattice2D
else:
    from .lattice import Lattice2D

__all__ = [
    "BrillouinZone2D"
]

class BrillouinZone2D:
    def __init__(
        self,
        lattice: Lattice2D,
        bz_sample_lattice: Lattice2D,
        eps: float=1e-12
    ):
        self.lattice = lattice
        self.bz_sample_lattice = bz_sample_lattice

        b1, b2 = lattice.reciprocal_lattice_vectors
        t1, t2 = lattice.lattice_vectors

        (n11, n12), offset1 = bz_sample_lattice.divmod(b1)
        (n21, n22), offset2 = bz_sample_lattice.divmod(b2)
        
        assert np.linalg.norm(offset1) < eps
        assert np.linalg.norm(offset2) < eps

        self.b1 = b1
        self.b2 = b2
        self.t1 = t1
        self.t2 = t2
        
        self.n11 = n11
        self.n12 = n12
        self.n21 = n21
        self.n22 = n22

        B = max(np.linalg.norm(b1), np.linalg.norm(b2))
        T = min(np.linalg.norm(t1), np.linalg.norm(t2))
        M = np.ceil(B / T).astype(int)

        xx = np.linspace(-M, M, 2 * M + 1)
        yy = np.linspace(-M, M, 2 * M + 1)
        mgrid = np.meshgrid(xx, yy)
        sample_coords, _ = pack(mgrid, "x y *")
        candidates = bz_sample_lattice.get_points(*mgrid, flatten=False)

        candidates = rearrange(candidates, "x y a -> (x y) a")
        sample_coords = rearrange(sample_coords, "x y a -> (x y) a")

        coords, _ = lattice.reciprocal_divmod(candidates)
        first_bz = np.logical_and(coords[:,0]==0, coords[:,1]==0)

        self.sample_positions = candidates[first_bz]
        self.sample_coords = sample_coords[first_bz]
        self.n_samples = np.sum(first_bz)

        # maps
        self.idx_from_coord = {(n1, n2) : idx for idx, (n1, n2) in enumerate(self.sample_coords)}

        # look up tables
        N_s = self.n_samples
        self.neg_table = np.empty((N_s,), dtype=int)
        self.sum_table = np.empty((N_s, N_s), dtype=int)
        self.sub_table = np.empty((N_s, N_s), dtype=int)
        
        # initialization
        for i in range(N_s):
            self.neg_table[i] = self._idx_neg(i)

        for i, j in itertools.product(range(N_s), repeat=2):
            self.sum_table[i, j] = self._idx_sum(i, j)

        for i, j in itertools.product(range(N_s), repeat=2):
            j_neg = self.neg_table[j]
            self.sub_table[i, j] = self.sum_table[i, j_neg]

    def fold_coord(self, coord):
        pos = self.bz_sample_lattice.pos_from_coord(coord)
        (c1, c2), _ = self.lattice.reciprocal_divmod(pos)
        # G = c1 b1 + c2 b2 = (c1 n11 + c2 n21) t1 + (c1 n12 + c2 n22) t2

        m1, m2 = coord
        new1 = m1 - (c1 * self.n11 + c2 * self.n21)
        new2 = m2 - (c1 * self.n12 + c2 * self.n22)

        return (new1, new2)

    def _idx_neg(self, idx):
        coord = self.sample_coords[idx]
        neg_coord = tuple(-x for x in coord)
        folded_coord = self.fold_coord(neg_coord)
        return self.idx_from_coord[folded_coord]

    def _idx_sum(self, idx1, idx2):
        coord1 = self.sample_coords[idx1]
        coord2 = self.sample_coords[idx2]
        sum_coord = tuple(x + y for x, y in zip(coord1, coord2))
        folded_coord = self.fold_coord(sum_coord)
        return self.idx_from_coord[folded_coord]
    
    def neg(self, idx):
        return self.neg_table[idx]
    
    def sum(self, idx1, idx2):
        return self.sum_table[idx1, idx2]
    
    def sub(self, idx1, idx2):
        return self.sub_table[idx1, idx2]

    def momentum_sectors(n_particles):
        pass


def test1():
    import matplotlib.pyplot as plt
    sqrt3 = np.sqrt(3)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    a_M = 1.0
    a1 = a_M * e2
    a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
    lattice = Lattice2D(np.stack([a1, a2]))
    b = (4 * np.pi) / (a_M * sqrt3)
    b1, b2 = lattice.reciprocal_lattice_vectors
    b3 = -(b1 + b2)
    
    # N = 27 grid
    t1 = (1 / 9) * (b1 - b2)
    t2 = (1 / 9) * (b1 - b3)

    sample_lattice = Lattice2D(np.stack([t1, t2]))

    # N = 28 grid
    p1 = b1 + t2 - t1
    normb1 = np.linalg.norm(b1)
    normp1 = np.linalg.norm(p1)
    distb1p1 = np.linalg.norm(b1 - p1)
    scale = normb1 / normp1
    rot = -np.arccos((normb1 ** 2 + normp1 ** 2 - distb1p1 ** 2) / (2 * normb1 * normp1))
    sample_lattice = sample_lattice.transformed(scale=scale, rot=rot)

    N = 3
    xx = np.linspace(-N, N, 2 * N + 1)
    yy = np.linspace(-N, N, 2 * N + 1)
    mgrid = np.meshgrid(xx, yy, sparse=True)
    points = lattice.reciprocal().get_points(*mgrid)

    bz = BrillouinZone2D(lattice, sample_lattice)
    print(f"{bz.n11=}, {bz.n12=}, {bz.n21=}, {bz.n22=}")
    print(f"{bz.n_samples}")

    N = 18
    xx = np.linspace(-N, N, 2 * N + 1)
    yy = np.linspace(-N, N, 2 * N + 1)
    mgrid = np.meshgrid(xx, yy, sparse=True)
    samples = sample_lattice.get_points(*mgrid)
    
    lattice_coord, _ = lattice.reciprocal_divmod(samples)
    colors = np.zeros((samples.shape[0], 3))
    colors[:, 0] = 0.15
    colors[:, 1] = 0.5
    colors[:, 2] = 0.5
    
    colors += lattice_coord[:, 0][:, None] * np.array([[0, 0.2, 0]])
    colors += lattice_coord[:, 1][:, None] * np.array([[0, 0.0, 0.2]])
    
    colors = np.clip(colors, 0.0, 1.0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.patch.set_facecolor('k')
    ax.scatter(samples[:, 0], samples[:, 1], s=75, c=colors)
    ax.scatter(points[:, 0], points[:, 1], c='r')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    plt.show()

def test2():
    import matplotlib.pyplot as plt
    sqrt3 = np.sqrt(3)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    a_M = 1.0
    a1 = a_M * e2
    a2 = a_M * ((-sqrt3 / 2) * e1 + (1 / 2) * e2)
    lattice = Lattice2D(np.stack([a1, a2]))
    b = (4 * np.pi) / (a_M * sqrt3)
    b1, b2 = lattice.reciprocal_lattice_vectors
    b3 = -(b1 + b2)
    
    # N = 27 grid
    t1 = (1 / 9) * (b1 - b2)
    t2 = (1 / 9) * (b1 - b3)

    sample_lattice = Lattice2D(np.stack([t1, t2]))

    # N = 28 grid
    p1 = b1 + t2 - t1
    normb1 = np.linalg.norm(b1)
    normp1 = np.linalg.norm(p1)
    distb1p1 = np.linalg.norm(b1 - p1)
    scale = normb1 / normp1
    rot = -np.arccos((normb1 ** 2 + normp1 ** 2 - distb1p1 ** 2) / (2 * normb1 * normp1))
    sample_lattice = sample_lattice.transformed(scale=scale, rot=rot)

    bz = BrillouinZone2D(lattice, sample_lattice)
    N_s = bz.n_samples
    
    # negation test
    print("testing double negation...")
    for i in range(N_s):
        assert bz.neg(bz.neg(i)) == i

    for i in range(N_s):
        neg_i = bz.neg(i)
        print(f"-({i}) = ({neg_i})")

    N = 30

    # summation test
    
    # commutativity test
    print("testing commutativity...")
    for i, j in itertools.product(range(N_s), repeat=2):
        assert bz.sum(i, j) == bz.sum(j, i)

    # associativity test
    print("testing associativity...")
    for i, j, k in itertools.product(range(N_s), repeat=3):
        assert bz.sum(i, bz.sum(j, k)) == bz.sum(bz.sum(i, j), k)


    ij = list(itertools.product(range(N_s), repeat=2))

    tests = random.sample(ij, N)
    for i, j in tests:
        print(f"({i}) + ({j}) = ({bz.sum(i, j)})")

    # subtraction test

    print("testing -(a - b) == b - a ...")
    for i, j in itertools.product(range(N_s), repeat=2):
        assert bz.neg(bz.sub(i, j)) == bz.sub(j, i)

    tests = random.sample(ij, N)
    for i, j in tests:
        print(f"({i}) - ({j}) = ({bz.sub(i, j)})")


    # plots


    # First BZ Boundary:
    sample_positions = bz.sample_positions
    R = 3 * np.linalg.norm(t1)
    thetas = np.linspace(np.pi / 2, 5 * np.pi / 2, 7)
    hexagon_x = R * np.cos(thetas)
    hexagon_y = R * np.sin(thetas)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    pos_x = sample_positions[:, 0]
    pos_y = sample_positions[:, 1]
    ax.scatter(pos_x, pos_y, s=250, c='k')
    for i in range(N_s):
        x = pos_x[i]
        y = pos_y[i]
        ax.text(x, y, f"{i}", fontsize=8, color="w", ha='center', va='center')

    ax.plot(hexagon_x, hexagon_y, color='k')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.show()

if __name__ == "__main__":
    test1()
