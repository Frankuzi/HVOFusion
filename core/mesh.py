import torch

class Mesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, colors, device='cpu'):
        self.device = device
        # 将mc后的顶点和面保存为torch类型
        self.vertices = vertices.to(device, dtype=torch.float32) if torch.is_tensor(vertices) else torch.tensor(vertices, dtype=torch.float32, device=device)
        self.indices = indices.to(device, dtype=torch.int64) if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.int64, device=device) if indices is not None else None
        self.colors = colors.to(device, dtype=torch.float32) if torch.is_tensor(colors) else torch.tensor(colors, dtype=torch.float32, device=device) if colors is not None else None

        if self.indices is not None:
            self.compute_normals()

        self._edges = None
        self._edges_mask = None
        self._verts_mask = None
        self._connected_faces = None
        self._laplacian = None
        self._planar = None
        self._planar_normals = None

    def to(self, device):
        mesh = Mesh(self.vertices.to(device), self.indices.to(device), self.colors.to(device), device=device)
        mesh._edges = self._edges.to(device) if self._edges is not None else None
        mesh._edges_mask = self._edges_mask.to(device) if self._edges_mask is not None else None
        mesh._verts_mask = self._verts_mask.to(device) if self._verts_mask is not None else None
        mesh._connected_faces = self._connected_faces.to(device) if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        mesh._planar = self._planar.to(device) if self._planar is not None else None
        mesh._planar_normals = self._planar_normals.to(device) if self._planar_normals is not None else None
        return mesh

    def detach(self):
        mesh = Mesh(self.vertices.detach(), self.indices.detach(), self.colors.detach(), device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._edges_mask = self._edges_mask.detach() if self._edges_mask is not None else None
        mesh._verts_mask = self._verts_mask.detach() if self._verts_mask is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        mesh._planar = self._planar.detach() if self._planar is not None else None
        mesh._planar_normals = self._planar_normals.detach() if self._planar_normals is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """

        assert len(vertices) == len(self.vertices)

        mesh_new = Mesh(vertices, self.indices, self.colors, self.device)        # 只更新顶点 面不发生变化
        mesh_new._edges = self._edges           # 仅改变顶点位置的时候 _edges _connected_faces _laplacian不会改变 因为indices拓扑结构不变
        mesh_new._edges_mask = self._edges_mask
        mesh_new._verts_mask = self._verts_mask
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        mesh_new._planar = self._planar
        mesh_new._planar_normals = self._planar_normals
        return mesh_new

    @property
    def edges(self):        # 计算三角面的各个边 并去除重复边
        if self._edges is None:
            from core.geometry import find_edges
            self._edges, self._edges_mask, self._verts_mask = find_edges(self.indices, self.vertices)      # [2558, 3] 输入面的三个顶点对应verts中的index
        return self._edges

    @property
    def connected_faces(self):
        if self._connected_faces is None:
            from core.geometry import find_connected_faces
            self._connected_faces = find_connected_faces(self.vertices, self.indices, self.face_normals)  # 找到每个边连接的两个面 [3837, 2] 第一维表示边序号
        return self._connected_faces

    @property
    def laplacian(self):
        if self._laplacian is None:
            from core.geometry import compute_laplacian_uniform
            self._laplacian = compute_laplacian_uniform(self)   # [1273, 1273]
        return self._laplacian
    
    @property
    def planar(self):
        if self._planar is None:
            import planar
            self._planar, self._planar_normals = planar.findPlanar(self.indices, self.vertex_normals, self.face_normals, 0.9999)
        return self._planar, self._planar_normals
            
    # 计算mesh的连接性 
    def compute_connectivity(self):
        self._edges = self.edges
        self._laplacian = self.laplacian
        self._connected_faces = self.connected_faces
        # self._planar, self._planar_normals = self.planar

    def compute_normals(self):
        # Compute the face normals
        # vertices[1273, 3] 顶点坐标
        # indices[2558, 3] 每个面对应的顶点index
        a = self.vertices[self.indices][:, 0, :]    # 取出三角面的三个顶点 [2558, 3]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1)   # 根据三角形计算三角形的法线 [2558, 3]
        '''
        计算三角面的法线可以使用torch中的cross函数。假设三角面的三个点分别为$A(x_1,y_1,z_1)$，$B(x_2,y_2,z_2)$和$C(x_3,y_3,z_3)$，则可以使用以下代码计算法线：

        import torch

        A = torch.tensor([x1, y1, z1])
        B = torch.tensor([x2, y2, z2])
        C = torch.tensor([x3, y3, z3])

        AB = B - A
        AC = C - A

        normal = torch.cross(AB, AC)
        '''

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals) # [1273, 3] 顶点法线
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals) # 每个顶点的法线为 该顶点所涉及的面的法线归一化后的结果
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)    # 对每个顶点法线进行归一化

def face_areas_normals(faces, vs):
    face_normals = torch.cross(vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
                               vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :], dim=2)
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / (face_areas[:, :, None] + 1e-8)
    face_areas = 0.5*face_areas
    return face_areas, face_normals

def sample_surface(faces, vs, count):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices
    faces: triangle faces (torch.long)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    normals: (count, 3) corresponding face normals for points
    """
    bsize, nvs, _ = vs.shape
    weights, normal = face_areas_normals(faces, vs)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(probs=weights / weights_sum[:, None])
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)

    return samples, normals