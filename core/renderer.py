import numpy as np
import nvdiffrast.torch as dr
import torch

class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.
    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    def __init__(self, device, near=1, far=1000):
        self.glctx = dr.RasterizeGLContext(device=device) 
        self.device = device
        self.near = near
        self.far = far

    def set_near_far(self, views, samples, epsilon=0.1):
        """ Automatically adjust the near and far plane distance
        """

        mins = []
        maxs = []
        for view in views:
            samples_projected = view.project(samples, depth_as_distance=True)   
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)        
        self.near = 0.01 # near - (near * epsilon)
        self.far = far + (far * epsilon)

    @staticmethod
    def transform_pos(mtx, pos):    
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)       
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
        projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                fy=camera.K[1,1],
                                                cx=camera.K[0,2],
                                                cy=camera.K[1,2],
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=camera.device)
        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t
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
            P = Renderer.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far)    
            pos_clip = Renderer.transform_pos(P, mesh.vertices)  
            idx = mesh.indices.int()   
            rast, rast_out_db = dr.rasterize(self.glctx, pos_clip, idx, resolution=view.resolution)  

            # Collect arbitrary output variables (aovs)
            if "mask" in channels:
                mask = torch.clamp(rast[..., -1:], 0, 1)    
                gbuffer["mask"] = dr.antialias(mask, rast, pos_clip, idx)[0] if with_antialiasing else mask[0]   

            if "position" in channels or "depth" in channels:
                position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)      
                gbuffer["position"] = dr.antialias(position, rast, pos_clip, idx)[0] if with_antialiasing else position[0]

            if "normal" in channels:
                normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)   
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
            P = Renderer.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far)    
            pos_clip = Renderer.transform_pos(P, mesh.vertices.detach())  
            idx = mesh.indices.int()    
            rast, rast_out_db = dr.rasterize(self.glctx, pos_clip, idx, resolution=view.resolution)  
            
            mask = torch.clamp(rast[..., -1:], 0, 1)    
            gbuffer["mask"] = dr.antialias(mask, rast, pos_clip, idx)[0] if with_antialiasing else mask[0]   
            
            position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)       
            gbuffer["position"] = dr.antialias(position, rast, pos_clip, idx)[0] if with_antialiasing else position[0]
            
            normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)   
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
            
