import numpy as np
import torch
from typing import List, Tuple, Union

from core.mesh import Mesh
from core.view import View

# from torch.utils.cpp_extension import load
# faceConnected = load(name="faceConnected", sources=["third_party/src/connected.cpp"], verbose=True)
import faceConnected

def find_edges(indices, vertices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face 
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2] 
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)
    
    edges_mask = vertex_mask = None
    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1) 
        edges, counts = torch.unique(edges, dim=0, return_counts=True)
        mask = counts == 1
        edges_indices = torch.nonzero(mask).squeeze()
        vertex_indices = edges[edges_indices]
        vertex_indices = torch.unique(torch.flatten(vertex_indices))
        edges_mask = torch.ones(edges.size(0), dtype=torch.bool, device=edges.device)
        edges_mask[edges_indices] = False
        vertex_mask = torch.ones(vertices.size(0), dtype=torch.bool, device=vertices.device)
        vertex_mask[vertex_indices] = False

    return edges, edges_mask, vertex_mask

def find_connected_faces(vertices, indices, normals):
    edges, _, _ = find_edges(indices, vertices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order 对每一行的两个元素进行排序，使得每一行的元素从小到大排列
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True) 
    # assert counts.max() == 2
    
    face_correspondences = faceConnected.computeFaceCorrespondences(inverse_indices.cpu(), counts.max(), counts.shape[0], indices.shape[0])
    return face_correspondences[counts == 2][:, :2].to(device=indices.device)

class AABB:
    def __init__(self, points):
        """ Construct the axis-aligned bounding box from a set of points.   从一组点构造轴对齐的包围框

        Args:
            points (tensor): Set of points (N x 3).
        """
        self.min_p, self.max_p = np.amin(points, axis=0), np.amax(points, axis=0)

    @classmethod
    def load(cls, path):
        points = np.loadtxt(path)
        return cls(points.astype(np.float32))

    def save(self, path):
        np.savetxt(path, np.array(self.minmax))

    @property
    def minmax(self):
        return [self.min_p, self.max_p]

    @property
    def center(self):
        return 0.5 * (self.max_p + self.min_p)

    @property
    def longest_extent(self): 
        return np.amax(self.max_p - self.min_p)

    @property
    def corners(self):      
        return np.array([
            [self.min_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.max_p[1], self.max_p[2]],
            [self.min_p[0], self.max_p[1], self.max_p[2]]
        ])
    
    @property
    def xyz_length(self):
        return np.array([self.max_p[0], self.max_p[1], self.max_p[2]]) - np.array([self.min_p[0], self.min_p[1], self.min_p[2]])

def normalize_aabb(aabb: AABB, side_length: float = 1):
    """ Scale and translate an axis-aligned bounding box to fit within a cube [-s/2, s/2]^3 centered at (0, 0, 0),
        with `s` the side length. 将AABB框进行标准化 中心点移动到0,0,0位置 边长归一化为1

    Args:
        aabb (AABB): The axis-aligned bounding box.
        side_length: Side length of the resulting cube. 

    Returns:
        Tuple of forward transformation A, that normalizes the bounding box and its inverse.
    """

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -aabb.center         

    s = side_length / aabb.longest_extent 
    S = np.diag([s, s, s, 1]).astype(dtype=np.float32)

    A = S @ T

    return A, np.linalg.inv(A)

def compute_laplacian_uniform(mesh):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """
    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    verts_packed = mesh.vertices 
    edges_packed = mesh.edges    
    V = mesh.vertices.shape[0]      

    e0, e1 = edges_packed.unbind(1)     

    idx01 = torch.stack([e0, e1], dim=1)  
    idx10 = torch.stack([e1, e0], dim=1)  
    idx = torch.cat([idx01, idx10], dim=0).t()  

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)    
    A = torch.sparse.FloatTensor(idx, ones, (V, V))     

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense() 

    # We construct the Laplacian matrix by adding the non diagonal values 
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge 
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)    
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)   
    val = torch.cat([deg0, deg1])       # [7674]
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1. 
    idx = torch.arange(V, device=mesh.device)
    idx = torch.stack([idx, idx], dim=0)        
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)    
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))   

    return L

def create_coordinate_grid(size: int, scale: float = 1.0, device: torch.device = 'cpu') -> torch.tensor:
    """ Create 3d grid of coordinates, [-scale, scale]^3. 生成3D网格坐标

    Args:
        size: Number of grid samples in each dimension.
        scale: Scaling factor applied to the grid coordinates.
        device: Device of the returned grid.

    Returns:
        Grid as tensor with shape (H, W, D, 3).
    """

    grid = torch.stack(torch.meshgrid(                      
        torch.linspace(-1.0, 1.0, size, device=device),
        torch.linspace(-1.0, 1.0, size, device=device),
        torch.linspace(-1.0, 1.0, size, device=device)
    ), dim=-1)

    return grid * scale

def marching_cubes(voxel_grid: torch.tensor, voxel_occupancy: torch.tensor, level: float = 0.5, **kwargs) -> Tuple[torch.tensor, torch.IntTensor]:
    """ Compute the marching cubes surface from an occupancy grid.

    Args:
        voxel_grid: Coordinates of the voxels with shape (HxWxDx3)
        voxel_occupancy: Occupancy of the voxels with shape (HxWxD), where 1 means occupied and 0 means not occupied.
        level: Occupancy value that marks the surface. 

    Returns:
        Array of vertices (Nx3) and face indices (Nx3) of the marching cubes surface.
    """

    from skimage import measure
    spacing = (voxel_grid[1, 1, 1] - voxel_grid[0, 0, 0]).cpu().numpy()
    vertices, faces, normals, values = measure.marching_cubes_lewiner(voxel_occupancy.cpu().numpy(), level=0.5, spacing=spacing, **kwargs)  

    # Re-center vertices 因为marching_cubes_lewiner
    vertices += voxel_grid[0, 0, 0].cpu().numpy()

    vertices = torch.from_numpy(vertices.copy()).to(voxel_grid.device)
    faces = torch.from_numpy(faces.copy()).to(voxel_grid.device)

    return vertices, faces

def sample_points_from_mesh(mesh, num_samples: int):

    # TODO: interpolate other mesh attributes
    # This code is adapted from from PyTorch3D
    with torch.no_grad():
        v0, v1, v2 = mesh.vertices[mesh.indices].unbind(1)
        areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

        sample_face_idxs = areas.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)

    # Get the vertex coordinates of the sampled faces.
    face_verts = mesh.vertices[mesh.indices]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        1, num_samples, mesh.vertices.dtype, mesh.vertices.device
    )
    samples = torch.cat([w0, w1, w2]).permute(1, 0)

    valid = sample_face_idxs < len(mesh.indices)
    indices = sample_face_idxs[valid]
    samples = samples[valid]

    sampled_faces = mesh.indices[indices]

    positions = torch.sum(mesh.vertices[sampled_faces] * samples.unsqueeze(-1), dim=-2)
    normals = torch.sum(mesh.vertices[sampled_faces] * samples.unsqueeze(-1), dim=-2)

    return positions, normals


def _rand_barycentric_coords(size1, size2, dtype: torch.dtype, device: torch.device):
    """
    # This code is taken from PyTorch3D
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # pyre-fixme[7]: Expected `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` but
    #  got `Tuple[float, typing.Any, typing.Any]`.
    return w0, w1, w2

def filter_mesh(v, f, m):
    f = f.to(dtype=torch.int64)

    # Create mapping from old vertex indices to new ones
    num_new_vertices = m.sum()
    old_to_new = torch.full((v.shape[0],), -1, device=v.device, dtype=f.dtype)
    old_to_new[m] = torch.arange(num_new_vertices, device=v.device, dtype=f.dtype)

    #
    v_new = v[m]
    f_new = torch.stack([
        old_to_new[f[..., 0]],
        old_to_new[f[..., 1]],
        old_to_new[f[..., 2]],
    ], dim=-1)
    f_new = f_new[torch.all(f_new != -1, dim=-1)]

    return v_new, f_new
