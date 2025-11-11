# from dataclasses import dataclass
from typing import overload, Optional, Union
import numpy as np
import warnings

from einops import einsum, rearrange, pack, unpack


# @dataclass
# class LatticeSite:
#     """Dataclass for single lattice site
    
#     Attributes:
#         id (int): integer ID of this site
#         position (np.ndarray): real-space position of this site
#         basis_coord (np.ndarray): basis coordinates of this site 
#     """

#     id: int
#     position: np.ndarray
#     basis_coord: np.ndarray

def _split_ij(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return coords[..., 0], coords[..., 1]

class Lattice2D:
    """A finite lattice
    """

    def __init__(
        self,
        lattice_vectors: np.ndarray,
        lengths: np.ndarray=None,
        # *,
        # offset: np.ndarray=None
    ):
        """Construct a lattice

        +
        """
        assert lattice_vectors.shape == (2, 2)
        assert lengths is None or lengths.shape == (2,)
        # assert offset is None or offset.shape == (2,)

        self._lattice_vectors = lattice_vectors
        self._ndim = 2
        self._lengths = lengths
        # self._offset = offset or np.zeros((self._ndim,))

        a0 = self._lattice_vectors[0]
        a1 = self._lattice_vectors[1]
        area = (a0[0] * a1[1] - a0[1] * a1[0])
        self._reciprocal_lattice_vectors = (2 * np.pi / area) * np.array([
            [a1[1], -a1[0]], 
            [-a0[1], a0[0]]
        ])

        self._candidate_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


    @property
    def lattice_vectors(self):
        return self._lattice_vectors
    
    @property
    def a1(self):
        return self._lattice_vectors[0]
    
    @property
    def a2(self):
        return self._lattice_vectors[1]

    @property
    def unit_cell_area(self):
        a1 = self._lattice_vectors[0]
        a2 = self._lattice_vectors[1]
        return np.abs(a1[0] * a2[1] - a1[1] * a2[0])
    
    @property
    def ndim(self):
        return self._ndim
    
    @property
    def lengths(self):
        return self._lengths
    
    # @property
    # def offset(self):
    #     return self._offset
    
    @property
    def reciprocal_lattice_vectors(self):
        """Reciprocal lattice vectors (b0, b1) as rows."""
        return self._reciprocal_lattice_vectors
    
    def transformed(self, *, scale=1.0, rot=0.0):
        """
        Applies scaling and rotation to the lattice vectors and returns a new 
        Lattice2D instance.
        
        Args:
            scale (float): The scaling factor.
            rotation (float): The rotation angle in radians.
            
        Returns:
            Lattice2D: A new instance with transformed lattice vectors.
        """
        c = np.cos(rot)
        s = np.sin(rot)

        rot_matrix_T = np.array([[c, s], 
                                 [-s, c]])
        transform_matrix_T = rot_matrix_T * scale
        transformed_vectors = self._lattice_vectors @ transform_matrix_T

        return Lattice2D(
            lattice_vectors=transformed_vectors,
            lengths=self._lengths
        )


    @staticmethod
    def _divmod(lvs, rlvs, cand, vec, precision):
        # vec_cent = vec - offset # vec = vec_cent + offset

        coeffs = einsum(vec, rlvs, '... j, i j -> ... i') / (2 * np.pi)

        floor_coords = np.floor(coeffs)

        candidate_coords = floor_coords[..., None, :] + cand
        candidate_lattice_pos = einsum(candidate_coords, lvs, '... c j, j i -> ... c i')
        dist_vectors = vec[..., None, :] - candidate_lattice_pos
        dist_sqs = np.sum(dist_vectors**2, axis=-1)

        dist_sqs_rounded = np.round(dist_sqs, decimals=precision)

        cand_x = candidate_coords[..., 0]
        cand_y = candidate_coords[..., 1]
        sort_keys = (cand_x, cand_y, dist_sqs_rounded)
        sorted_indices = np.lexsort(sort_keys, axis=-1)
        min_index = sorted_indices[..., 0]
        min_index_expanded = min_index[..., None, None]

        closest_lattice_coord_batch = np.take_along_axis(candidate_coords, min_index_expanded, axis=-2)
        wigner_seitz_vec_batch = np.take_along_axis(dist_vectors, min_index_expanded, axis=-2)
        
        closest_lattice_coord = np.squeeze(closest_lattice_coord_batch, axis=-2)
        wigner_seitz_vec = np.squeeze(wigner_seitz_vec_batch, axis=-2)

        return closest_lattice_coord.astype(int), wigner_seitz_vec
    

    def divmod(self, vec, precision=12):
        """
        Divide the vector into lattice coordinate and offset on Wigner-Seitz cell.
        
        A vector 'vec' is decomposed as:
        vec = (n0 * a0 + n1 * a1) + wigner_seitz_vec + offset
        
        where (n0, n1) are integer lattice coordinates and wigner_seitz_vec
        is the smallest vector connecting a lattice point to vec_cent.
        
        Returns:
            tuple: (lattice_coords, wigner_seitz_vec)
                lattice_coords (np.ndarray): Integer array (n0, n1)
                wigner_seitz_vec (np.ndarray): Vector offset in the Wigner-Seitz cell
        """
        return self._divmod(
            self.lattice_vectors, 
            self.reciprocal_lattice_vectors,
            # self.offset,
            self._candidate_offsets,
            vec,
            precision
        )

    def reciprocal_divmod(self, vec, precision=12):
        """
        Divide the vector into reciprocal lattice coordinate and offset on First Brillouin zone.
        
        A vector 'vec' is decomposed as:
        vec = (n0 * a0 + n1 * a1) + wigner_seitz_vec + offset
        
        where (n0, n1) are integer lattice coordinates and wigner_seitz_vec
        is the smallest vector connecting a lattice point to vec_cent.
        
        Returns:
            tuple: (lattice_coords, wigner_seitz_vec)
                lattice_coords (np.ndarray): Integer array (n0, n1)
                wigner_seitz_vec (np.ndarray): Vector offset in the Wigner-Seitz cell
        """
        return self._divmod(
            self.reciprocal_lattice_vectors, 
            self.lattice_vectors,
            # np.zeros((self._ndim,)),
            self._candidate_offsets,
            vec,
            precision
        )
    
    def pos_from_coord(self, coord: tuple[int, int]):
        n1, n2 = coord
        a1, a2 = self.lattice_vectors
        return n1 * a1 + n2 * a2

    def fold_to_bz1(self, vec: np.ndarray):
        _, offset = self.reciprocal_divmod(vec)
        return offset

    def reciprocal(self):
        return Lattice2D(self.reciprocal_lattice_vectors)

    @overload
    def get_points(self, coords: np.ndarray) -> np.ndarray: ...

    # Case 2: 인자가 2개인 경우 (flatten은 '*' 뒤에 있어 키워드 전용으로 강제됨)
    @overload
    def get_points(self, xgrid: np.ndarray, ygrid: np.ndarray, *, flatten: bool = False) -> np.ndarray: ...
    
    def get_points(self, arg1: np.ndarray, arg2: Union[np.ndarray, bool, None] = None, *, flatten: Optional[bool] = None):
        xgrid: np.ndarray
        ygrid: np.ndarray
        final_flatten: bool = False
        
        if isinstance(arg2, np.ndarray):
            xgrid, ygrid = arg1, arg2
            # 이 경우 flatten은 반드시 키워드 인자로 들어와야 함
            if flatten is not None:
                final_flatten = flatten
        else:
            xgrid, ygrid = _split_ij(arg1)
            if arg2 is not None or flatten is not None:
                warnings.warn("Single input detected: 'flatten' argument is ignored.", UserWarning)
            
        a1, a2 = self.lattice_vectors
        points = xgrid[..., None] * a1[None, None, :] + ygrid[..., None] * a2[None, None, :]
        if flatten:
            return rearrange(points, "x y a -> (x y) a")
        else:
            return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sqrt3 = np.sqrt(3)
    tri_lattice = Lattice2D(np.array(
        [[1/2, -sqrt3/2], [1/2, sqrt3/2]]
    ))


    N = 3
    xx = np.linspace(-N, N, 2 * N + 1)
    yy = np.linspace(-N, N, 2 * N + 1)
    mgrid = np.meshgrid(xx, yy, sparse=True)
    points = tri_lattice.get_points(*mgrid, flatten=True)

    A = tri_lattice.lattice_vectors
    t1 = (2 * A[0] + A[1]) / 2
    t2 = (A[0] + 2 * A[1]) / 2

    eps = 1e-6
    fine_points = np.arange(-N, N + eps, 0.05)[:, None, None] * t1[None, None, :] + np.arange(-N, N + eps, 0.05)[None, :, None] * t2[None, None, :]
    fine_points = rearrange(fine_points, "x y a -> (x y) a")

    lattice_coord, offsets = tri_lattice.divmod(fine_points)
    colors = np.full((fine_points.shape[0], 3), 0.5)
    colors += lattice_coord[:, 0][:, None] * np.array([[0.15, 0.0, 0]])
    colors += lattice_coord[:, 1][:, None] * np.array([[0, 0.15, 0.15]])
    
    colors = np.clip(colors, 0.0, 1.0)

    plt.figure(figsize=(8, 8))
    plt.scatter(fine_points[:, 0], fine_points[:, 1], s=5, c=colors)
    plt.scatter(points[:, 0], points[:, 1], c='k')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()