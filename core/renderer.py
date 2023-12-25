import numpy as np
import nvdiffrast.torch as dr
import torch

class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.
    基于栅格化的三角形网格渲染器 为一组视图生成G-buffers

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    def __init__(self, device, near=1, far=1000):
        self.glctx = dr.RasterizeGLContext(device=device)    # 用于创建一个OpenGL上下文，以便在OpenGL中进行光栅化
        self.device = device
        self.near = near
        self.far = far

    def set_near_far(self, views, samples, epsilon=0.1):
        """ Automatically adjust the near and far plane distance
            根据所有的views来计算远点和近点
            通过AABB的八个顶点信息根据view相机参数投影到屏幕上
        """

        mins = []
        maxs = []
        for view in views:
            samples_projected = view.project(samples, depth_as_distance=True)   # 将空间点投影到屏幕上 samples_projected的最后一维保存了 空间点距离 depth_as_distance=True表示将距离norm后的实际距离
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)        # 在所有views的距离中选择最近和最远的距离
        self.near = 0.01 # near - (near * epsilon)     # 稍微扩大一点
        self.far = far + (far * epsilon)

    @staticmethod
    def transform_pos(mtx, pos):    # mtx 表示投影矩阵 pos表示空间点坐标
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)       # 变换为齐次坐标用于和[4,4]矩阵相乘
        return torch.matmul(posw, t_mtx.t())[None, ...]

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        https://pengfeixc.com/blogs/computer-graphics/3D-matrix-transformation-part-three
        """
        return torch.tensor([[2.0*fx/width,           0,       (2.0 * cx / width) - 1,                  0],
                            [         0, 2.0*fy/height,      (2.0 * cy / height) - 1,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device) 
    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000):
        # 创建渲染的投影矩阵 根据内参计算
        projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                fy=camera.K[1,1],
                                                cx=camera.K[0,2],
                                                cy=camera.K[1,2],
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=camera.device)
        # 根据当前view视角的相机外参构造R矩阵
        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t
        # 将外参变换到opengl的旋转矩阵格式
        gl_transform = torch.tensor([[1., 0,  0,  0],
                                    [0,  1., 0,  0],
                                    [0,  0, -1., 0],
                                    [0,  0,  0,  1.]], device=camera.device)
        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    def render(self, views, mesh, channels, with_antialiasing=True):
        """ Render G-buffers from a set of views.

        Args:z
            views (List[Views]): 视角信息
            mesh
        """

        # TODO near far should be passed by view to get higher resolution in depth
        gbuffers = []
        for i, view in enumerate(views):
            gbuffer = {}

            # Rasterize only once
            P = Renderer.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far)    # 计算投影矩阵 将3D空间点投影到2D屏面上
            pos_clip = Renderer.transform_pos(P, mesh.vertices)  # [1, 1273, 4] 顶点信息 4维向量 xyza 后续要归一化
            idx = mesh.indices.int()    # [2558, 3] 面信息
            # rasterize函数的用法是将三角面片渲染成深度和导数张量 
            # pos 4维向量 xyza 后续要归一化 idx为每个面的index指示每个面在pos中的序号 rasr输出为 (u, v, z/w, triangle_id) https://nvlabs.github.io/nvdiffrast/ 
            # 这里的u,v并不是点坐标而是 在某一个三角面关于三个顶点的偏移量  第一个顶点(u, v) =(1,0)，第二个顶点(u, v) =(0,1)，第三个顶点(u, v) =(0,0)
            # triangle_id是三角形索引，偏移1。没有三角形栅格化的像素将在所有通道中为0。
            rast, rast_out_db = dr.rasterize(self.glctx, pos_clip, idx, resolution=view.resolution)  # [1, 1200, 1600, 4]

            # Collect arbitrary output variables (aovs)
            if "mask" in channels:
                mask = torch.clamp(rast[..., -1:], 0, 1)    # 根据triangle_id筛选有光线穿透的有效三角形（即该反省位置可以看到的面）
                gbuffer["mask"] = dr.antialias(mask, rast, pos_clip, idx)[0] if with_antialiasing else mask[0]   # 进行抗锯齿

            if "position" in channels or "depth" in channels:
                position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)       # 根据rast中的 triangle_id找到对应三角形，再通过idx找到vertices对应的顶点坐标 最后通过rast中的uv进行插值
                gbuffer["position"] = dr.antialias(position, rast, pos_clip, idx)[0] if with_antialiasing else position[0]

            if "normal" in channels:
                normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)   # 同上 只不过是对顶点法向量进行插值
                gbuffer["normal"] = dr.antialias(normal, rast, pos_clip, idx)[0] if with_antialiasing else normal[0]

            if "depth" in channels:
                depth = view.project(mesh.vertices, depth_as_distance=False)[..., 2:3].contiguous()
                cdepth, _ = dr.interpolate(depth[None, ...], rast, idx)
                gbuffer["depth"] = dr.antialias(cdepth, rast, pos_clip, idx)[0] if with_antialiasing else cdepth[0]

            gbuffers += [gbuffer]

        return gbuffers

    def render_color(self, views, mesh, valbedo, vsh, with_antialiasing = True):
        gbuffers = []
        for i, view in enumerate(views):
            gbuffer = {}

            # Rasterize only once
            P = Renderer.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far)    # 计算投影矩阵 将3D空间点投影到2D屏面上
            pos_clip = Renderer.transform_pos(P, mesh.vertices.detach())  # [1, 1273, 4] 顶点信息 4维向量 xyza 后续要归一化
            idx = mesh.indices.int()    # [2558, 3] 面信息
            rast, rast_out_db = dr.rasterize(self.glctx, pos_clip, idx, resolution=view.resolution)  # [1, 1200, 1600, 4]
            
            mask = torch.clamp(rast[..., -1:], 0, 1)    # 根据triangle_id筛选有光线穿透的有效三角形（即该反省位置可以看到的面）
            gbuffer["mask"] = dr.antialias(mask, rast, pos_clip, idx)[0] if with_antialiasing else mask[0]   # 进行抗锯齿
            
            position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)       # 根据rast中的 triangle_id找到对应三角形，再通过idx找到vertices对应的顶点坐标 最后通过rast中的uv进行插值
            gbuffer["position"] = dr.antialias(position, rast, pos_clip, idx)[0] if with_antialiasing else position[0]
            
            normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)   # 同上 只不过是对顶点法向量进行插值
            gbuffer["normal"] = dr.antialias(normal, rast, pos_clip, idx)[0] if with_antialiasing else normal[0]
            
            depth = view.project(mesh.vertices, depth_as_distance=False)[..., 2:3].contiguous()
            cdepth, _ = dr.interpolate(depth[None, ...], rast, idx)
            gbuffer["depth"] = dr.antialias(cdepth, rast, pos_clip, idx)[0] if with_antialiasing else cdepth[0]
            
            sh, _ = dr.interpolate(vsh[None, ...], rast, idx)
            gbuffer["sh"] = dr.antialias(sh, rast, pos_clip, idx)[0] if with_antialiasing else sh[0]
            
            albedo, _ = dr.interpolate(valbedo[None, ...], rast, idx)
            gbuffer["albedo"] = dr.antialias(albedo, rast, pos_clip, idx)[0] if with_antialiasing else albedo[0]
            
            gbuffers += [gbuffer]

        return gbuffers, rast.squeeze()[..., 3]
            