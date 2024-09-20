import cv2
import re
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation

from core.camera import Camera

class View:
    """ A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, color, gray, mask, depth, camera, device='cpu'):
        self.color = color.to(device)
        self.gray = gray.to(device)
        self.mask = mask.to(device)
        self.depth = depth.to(device)
        self.camera = camera.to(device)
        self.device = device

    @classmethod
    def replica_load(cls, image_path, depth_path, device='cpu'):
        """ Load a view from a given image path.

        The paths of the camera matrices are deduced from the image path. 
        Given an image path `path/to/directory/foo.png`, the paths to the camera matrices
        in numpy readable text format are assumed to be `path/to/directory/foo_k.txt`, 
        `path/to/directory/foo_r.txt`, and `path/to/directory/foo_t.txt`.

        Args:
            image_path (Union[Path, str]): Path to the image file that contains the color and optionally the mask
            device (torch.device): Device where the images and camera are stored
        """

        image_path = Path(image_path)
        depth_path = Path(depth_path)
        
        K = np.loadtxt(image_path.parent / (image_path.stem + "_k.txt"))    
        R = np.loadtxt(image_path.parent / (image_path.stem + "_r.txt"))    
        t = np.loadtxt(image_path.parent / (image_path.stem + "_t.txt"))    
        R_inv = np.transpose(R)
        t_inv = np.dot(-R_inv, t)
        camera = Camera(K, R_inv, t_inv)
        
        # Load the color and grayscale
        image = Image.open(image_path)
        color = torch.FloatTensor(np.array(image))     
        color /= 255.0
        gray = torch.FloatTensor(np.array(transforms.Grayscale()(image)))
        gray /= 255.0
        
        # Load the depth
        depth = torch.FloatTensor(np.array(Image.open(depth_path)))     
        depth /= 6553.5
        depth[depth > 10] = 0           
        depth.unsqueeze_(-1)
        
        mask = torch.ones_like(color[:, :, 0:1])
        mask[depth == 0] = 0
        
        return cls(color, gray, mask, depth, camera, device=device)  
    
    @classmethod
    def scannet_load(cls, image_path, depth_path, device='cpu'):
        image_path = Path(image_path)
        depth_path = Path(depth_path)
        
        # Load the camera
        K = np.loadtxt(image_path.parent / (image_path.stem + "_k.txt"))    
        R = np.loadtxt(image_path.parent / (image_path.stem + "_r.txt"))    
        t = np.loadtxt(image_path.parent / (image_path.stem + "_t.txt"))    
        R_inv = np.transpose(R)
        t_inv = np.dot(-R_inv, t)
        camera = Camera(K, R_inv, t_inv)

         # Load the color and grayscale
        image = Image.open(image_path)
        image = cv2.imread(str(image_path), -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, (int(gray.shape[0] / 1.5), int(gray.shape[1] / 1.5)), cv2.INTER_AREA)
        gray = torch.FloatTensor(np.array(gray))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (int(image.shape[0] / 1.5), int(image.shape[1] / 1.5)), cv2.INTER_AREA)
        color = torch.FloatTensor(np.array(image))     
        color /= 255.0
        gray /= 255.0
        
        # Load the depth
        depth = torch.FloatTensor(np.array(Image.open(depth_path)))     
        depth /= 1000.0
        depth[depth > 10] = 0           
        depth.unsqueeze_(-1)
        
        mask = torch.ones_like(color[:, :, 0:1])
        mask[depth == 0] = 0

        return cls(color, gray, mask, depth, camera, device=device)  
    
    @classmethod
    def load2(cls, image_path, depth_path, mask_path, device='cpu'):
        """ Load a view from a given image path.

        The paths of the camera matrices are deduced from the image path. 
        Given an image path `path/to/directory/foo.png`, the paths to the camera matrices
        in numpy readable text format are assumed to be `path/to/directory/foo_k.txt`, 
        `path/to/directory/foo_r.txt`, and `path/to/directory/foo_t.txt`.

        Args:
            image_path (Union[Path, str]): Path to the image file that contains the color and optionally the mask
            device (torch.device): Device where the images and camera are stored
        """
        
        image_path = Path(image_path)
        depth_path = Path(depth_path)
        mask_path = Path(mask_path)
        
        # Load the camera 
        K = np.loadtxt(image_path.parent / ("cam" + re.findall(r'\d+', str(image_path))[0] + "_k.txt"))    
        R = np.loadtxt(image_path.parent / ("cam" + re.findall(r'\d+', str(image_path))[0] + "_r.txt"))    
        t = np.loadtxt(image_path.parent / ("cam" + re.findall(r'\d+', str(image_path))[0] + "_t.txt"))    
        R_inv = np.transpose(R)
        t_inv = np.dot(-R_inv, t)
        camera = Camera(K, R_inv, t_inv)
        
        # Load the color
        image = Image.open(image_path)
        dimage = image.resize((256, 192))
        color = torch.FloatTensor(np.array(dimage))     
        color /= 255.0
        
        # Load the depth
        depth = torch.FloatTensor(np.array(Image.open(depth_path)))     
        depth /= 1000.0
        depth[depth > 10] = 0           
        depth.unsqueeze_(-1)
        
        # mask = torch.FloatTensor(np.array(Image.open(mask_path))).unsqueeze(-1)     
        mask = torch.ones_like(color[:, :, 0:1])
        mask[depth == 0] = 0
        
        return cls(color, mask, depth, camera, device=device)  

    def to(self, device: str = "cpu"):
        self.color = self.color.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        self.camera = self.camera.to(device)
        self.device = device
        return self

    @property
    def resolution(self):
        return (self.color.shape[0], self.color.shape[1])
    
    def scale(self, inverse_factor):
        """ Scale the view by a factor.
        
        This operation is NOT differentiable in the current state as 
        we are using opencv.

        Args:
            inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
        """
        
        scaled_height = self.color.shape[0] // inverse_factor
        scaled_width = self.color.shape[1] // inverse_factor

        scale_x = scaled_width / self.color.shape[1]
        scale_y = scaled_height / self.color.shape[0]
        
        self.color = torch.FloatTensor(cv2.resize(self.color.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)
        if self.mask is not None:
            self.mask = torch.FloatTensor(cv2.resize(self.mask.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)).to(self.device)
            self.mask = self.mask.unsqueeze(-1) # Make sure the mask is HxWx1
        if self.depth is not None:
            torch.FloatTensor(cv2.resize(self.depth.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)

        self.camera.K = torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device) @ self.camera.K  

    def transform(self, A, A_inv=None):
        """ Transform the view pose with an affine mapping.
        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv. 
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix. 
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t    
        P = torch.zeros((3, 4), device=self.device)
        P[:3, :3] = self.camera.K @ R
        P[:3, 3] = self.camera.K @ t
        K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())   
        c = c[:3, 0] / c[3]     
        t = - R @ c

        # ensure unique scaling of K matrix
        K = K / K[2,2]
        
        self.camera.K = torch.from_numpy(K).to(self.device)
        self.camera.R = torch.from_numpy(R).to(self.device)
        self.camera.t = torch.from_numpy(t).to(self.device)
        
    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t). 将空间点坐标投射到对应view方向的图像平面上

        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """

        points_c = points @ torch.transpose(self.camera.R, 0, 1) + self.camera.t        
        pixels = points_c @ torch.transpose(self.camera.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]      
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)    
        return torch.cat([pixels, depths], dim=-1)      
    
    def unproject(self, depth):
        h = depth.size(0)
        w = depth.size(1)
        cx = self.camera.K[0][2].item()
        cy = self.camera.K[1][2].item()
        fx = self.camera.K[0][0].item()
        fy = self.camera.K[1][1].item()
        
        grids = torch.meshgrid([torch.arange(w, dtype=torch.float32, device=self.device), torch.arange(h, dtype=torch.float32, device=self.device)])
        u = grids[0].t().flatten()
        v = grids[1].t().flatten()

        Zc = depth.view(-1)
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        Oc = torch.ones(h * w, device=self.device)
        camera_coords = torch.stack([Xc, Yc, Zc, Oc])

        world_coords = self.camera.Rt.mm(camera_coords)
        points = world_coords.t()[:, :3]

        return points
    
class ViewDepth:
    """ A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, depth, camera, device='cpu'):
        self.depth = depth.to(device)
        self.camera = camera.to(device)
        self.device = device
    
    @classmethod
    def load(cls, depth_path, index, device='cpu'):
        depth_path = Path(depth_path)
        
        with open(depth_path.parent / "poseDepth.txt", 'r') as f:
            lines = f.readlines()
            pose = [float(i) for i in lines[index].split()]
        pose = np.array(pose).reshape((4, 4))
        R = pose[:3, :3]
        t = pose[:3, 3]
        R_inv = np.transpose(R)
        t_inv = np.dot(-R_inv, t)
        K = np.loadtxt(depth_path.parent / "depth_intrinsic.txt")    
        camera = Camera(K, R_inv, t_inv)
        
        # Load the depth
        depth = torch.FloatTensor(np.array(Image.open(depth_path)))     
        depth /= 1000.0
        depth[depth > 10] = 0           
        depth.unsqueeze_(-1)

        return cls(depth, camera, device=device)  

    def to(self, device: str = "cpu"):
        self.depth = self.depth.to(device)
        self.camera = self.camera.to(device)
        self.device = device
        return self

class ViewColor:
    """ A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, color, gray, mask, camera, device='cpu'):
        self.color = color.to(device)
        self.gray = gray.to(device)
        self.mask = mask.to(device)
        self.camera = camera.to(device)
        self.device = device

    @classmethod
    def load(cls, image_path, index, device='cpu'):
        image_path = Path(image_path)
        
        with open(image_path.parent / "poseColor.txt", 'r') as f:
            lines = f.readlines()
            pose = [float(i) for i in lines[index].split()]
        pose = np.array(pose).reshape((4, 4))
        R = pose[:3, :3]
        t = pose[:3, 3]
        R_inv = np.transpose(R)
        t_inv = np.dot(-R_inv, t)
        K = np.loadtxt(image_path.parent / "color_intrinsic.txt")    
        camera = Camera(K, R_inv, t_inv)

         # Load the color and grayscale
        image = Image.open(image_path)
        image = cv2.imread(str(image_path), -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = torch.FloatTensor(np.array(gray))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color = torch.FloatTensor(np.array(image))
        color /= 255.0
        gray /= 255.0
        
        mask = torch.ones_like(color[:, :, 0:1])

        return cls(color, gray, mask, camera, device=device)  
    
    def to(self, device: str = "cpu"):
        self.color = self.color.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        self.camera = self.camera.to(device)
        self.device = device
        return self

    @property
    def resolution(self):
        return (self.color.shape[0], self.color.shape[1])
    
    def scale(self, inverse_factor):
        """ Scale the view by a factor.
        
        This operation is NOT differentiable in the current state as 
        we are using opencv.

        Args:
            inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
        """
        
        scaled_height = self.color.shape[0] // inverse_factor
        scaled_width = self.color.shape[1] // inverse_factor

        scale_x = scaled_width / self.color.shape[1]
        scale_y = scaled_height / self.color.shape[0]
        
        self.color = torch.FloatTensor(cv2.resize(self.color.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)
        if self.mask is not None:
            self.mask = torch.FloatTensor(cv2.resize(self.mask.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)).to(self.device)
            self.mask = self.mask.unsqueeze(-1) # Make sure the mask is HxWx1
        if self.depth is not None:
            torch.FloatTensor(cv2.resize(self.depth.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)

        self.camera.K = torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device) @ self.camera.K  

    def transform(self, A, A_inv=None):
        """ Transform the view pose with an affine mapping.
        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv. 
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix. 
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t    
        P = torch.zeros((3, 4), device=self.device)
        P[:3, :3] = self.camera.K @ R
        P[:3, 3] = self.camera.K @ t
        K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())   
        c = c[:3, 0] / c[3]     
        t = - R @ c

        # ensure unique scaling of K matrix
        K = K / K[2,2]
        
        self.camera.K = torch.from_numpy(K).to(self.device)
        self.camera.R = torch.from_numpy(R).to(self.device)
        self.camera.t = torch.from_numpy(t).to(self.device)
        
    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t). 

        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """

        points_c = points @ torch.transpose(self.camera.R, 0, 1) + self.camera.t        
        pixels = points_c @ torch.transpose(self.camera.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]      
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)    
        return torch.cat([pixels, depths], dim=-1)      
    
    def unproject(self, depth):
        h = depth.size(0)
        w = depth.size(1)
        cx = self.camera.K[0][2].item()
        cy = self.camera.K[1][2].item()
        fx = self.camera.K[0][0].item()
        fy = self.camera.K[1][1].item()
        
        grids = torch.meshgrid([torch.arange(w, dtype=torch.float32, device=self.device), torch.arange(h, dtype=torch.float32, device=self.device)])
        u = grids[0].t().flatten()
        v = grids[1].t().flatten()

        Zc = depth.view(-1)
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        Oc = torch.ones(h * w, device=self.device)
        camera_coords = torch.stack([Xc, Yc, Zc, Oc])

        world_coords = self.camera.Rt.mm(camera_coords)
        points = world_coords.t()[:, :3]

        return points

class ViewPly:
    """ A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, color, gray, points, mask, camera, lidar, device='cpu'):
        self.color = color.to(device)
        self.gray = gray.to(device)
        self.point = points.to(device)
        self.mask = mask.to(device)
        self.camera = camera.to(device)
        self.lidar = lidar.to(device)
        self.device = device
    
    @classmethod
    def load(cls, image_path, ply_path, device='cpu'):
        image_path = Path(image_path)
        ply_path = Path(ply_path)
        
        # Load the camera 
        K = np.loadtxt(image_path.parent / (image_path.stem + "_k.txt"))    
        R = np.loadtxt(image_path.parent / (image_path.stem + "_r.txt"))    
        t = np.loadtxt(image_path.parent / (image_path.stem + "_t.txt"))    
        lidar = Camera(K, R, t)
        # r1 = Rotation.from_quat(np.array([0, 0, 0.924, 0.383]))
        # r2 = Rotation.from_quat(np.array([ 0.        ,  0.        , -0.95892427,  0.28366219]))
        # R = R @ r1.as_matrix() @ r2.as_matrix()
        view_transform = np.array([[1., 0, 0],
                                    [0, 0, 1.],
                                    [0, -1., 0]])
        r1 = Rotation.from_quat(np.array([0.        , 0.        ,  0.93200787, 0.36243804])).as_matrix() 
        R = r1 @ view_transform @ R     
        t -= np.array([-0.084, -0.025, 0.050])
        R_inv = np.transpose(R)
        t_inv = np.dot(-R_inv, t)
        camera = Camera(K, R_inv, t_inv)

        # Load the ply points by open3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = torch.from_numpy(np.asarray(pcd.points)).float()

        # Load the color and grayscale
        image = Image.open(image_path)
        color = torch.FloatTensor(np.array(image))
        color /= 255.0
        gray = torch.FloatTensor(np.array(transforms.Grayscale()(image)))
        gray /= 255.0

        mask = torch.ones_like(color[:, :, 0:1])

        return cls(color, gray, points, mask, camera, lidar, device=device)

    def to(self, device: str = "cpu"):
        self.color = self.color.to(device)
        self.gray = self.gray.to(device)
        self.point = self.point.to(device)
        self.camera = self.camera.to(device)
        self.lidar = self.lidar.to(device)
        self.device = device
        return self

    @property
    def resolution(self):
        return (self.color.shape[0], self.color.shape[1])
    
    def scale(self, inverse_factor):
        """ Scale the view by a factor.
        
        This operation is NOT differentiable in the current state as 
        we are using opencv.

        Args:
            inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
        """
        
        scaled_height = self.color.shape[0] // inverse_factor
        scaled_width = self.color.shape[1] // inverse_factor

        scale_x = scaled_width / self.color.shape[1]
        scale_y = scaled_height / self.color.shape[0]
        
        self.color = torch.FloatTensor(cv2.resize(self.color.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)
        # if self.mask is not None:
        #     self.mask = torch.FloatTensor(cv2.resize(self.mask.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)).to(self.device)
        #     self.mask = self.mask.unsqueeze(-1) # Make sure the mask is HxWx1
        # if self.depth is not None:
        #     torch.FloatTensor(cv2.resize(self.depth.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)

        self.camera.K = torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device) @ self.camera.K  

    def transform(self, A, A_inv=None):
        """ Transform the view pose with an affine mapping.
        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv. 
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix.
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t    
        P = torch.zeros((3, 4), device=self.device)
        P[:3, :3] = self.camera.K @ R
        P[:3, 3] = self.camera.K @ t
        K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())   
        c = c[:3, 0] / c[3]     
        t = - R @ c

        # ensure unique scaling of K matrix
        K = K / K[2,2]
        
        self.camera.K = torch.from_numpy(K).to(self.device)
        self.camera.R = torch.from_numpy(R).to(self.device)
        self.camera.t = torch.from_numpy(t).to(self.device)
        
    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t). 将空间点坐标投射到对应view方向的图像平面上

        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """

        points_c = points @ torch.transpose(self.camera.R, 0, 1) + self.camera.t        
        pixels = points_c @ torch.transpose(self.camera.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]      
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)    
        return torch.cat([pixels, depths], dim=-1)     
    
    def unproject(self, depth):
        h = depth.size(0)
        w = depth.size(1)
        cx = self.camera.K[0][2].item()
        cy = self.camera.K[1][2].item()
        fx = self.camera.K[0][0].item()
        fy = self.camera.K[1][1].item()
        
        grids = torch.meshgrid([torch.arange(w, dtype=torch.float32, device=self.device), torch.arange(h, dtype=torch.float32, device=self.device)])
        u = grids[0].t().flatten()
        v = grids[1].t().flatten()

        Zc = depth.view(-1)
        Xc = (u - cx) * Zc / fx
        Yc = (v - cy) * Zc / fy
        Oc = torch.ones(h * w, device=self.device)
        camera_coords = torch.stack([Xc, Yc, Zc, Oc])

        world_coords = self.camera.Rt.mm(camera_coords)
        points = world_coords.t()[:, :3]

        return points
