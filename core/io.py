import numpy as np
from pathlib import Path
import trimesh
import os
import glob
from natsort import natsorted

from core.mesh import Mesh
from core.view import View, ViewDepth, ViewColor, ViewPly

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    colors = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)
    if hasattr(mesh_, 'colors'):
        colors = np.array(mesh_.colors, dtype=np.float32)

    return Mesh(vertices, indices, colors, device)

def write_mesh(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None
    colors = (mesh.colors.numpy() * 255).astype('uint8') if mesh.colors is not None else None
    mesh_ = trimesh.Trimesh(vertices=vertices, faces=indices, vertex_colors=colors, process=False)
    mesh_.export(path, file_type='ply')

def read_views(directory, name, device):
    directory = Path(directory)
    if name == 'Replica':
        image_paths_jpg = natsorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.jpg')])      
        depth_paths_png = natsorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.png')])      
        
        views = []
        for image_path, depth_path in zip(image_paths_jpg, depth_paths_png):
            views.append(View.replica_load(image_path, depth_path, device))
        print("Found {:d} views".format(len(views)))
        return views
    if name == 'Scannet':
        depth_paths_png = natsorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.png')])      

        depth_views = []
        for i in range(len(depth_paths_png)):
            depth_path = depth_paths_png[i]
            depth_views.append(ViewDepth.load(depth_path, i, device))
        print("Found {:d} depth views".format(len(depth_views)))

        color_paths_jpg = natsorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.jpg')])      

        color_views = []
        for i in range(len(color_paths_jpg)):
            color_path = color_paths_jpg[i]
            color_views.append(ViewColor.load(color_path, i, device))
        print("Found {:d} color views".format(len(color_views)))
        return depth_views, color_views
    if name == 'Pcd':
        image_paths_png = natsorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.png')])
        pcd_paths_ply = natsorted([path for path in directory.iterdir() if (path.is_file() and path.suffix == '.ply')])
        views = []
        for image_path, pcd_paths in zip(image_paths_png, pcd_paths_ply):
            views.append(ViewPly.load(image_path, pcd_paths, device))
        print("Found {:d} views".format(len(views)))
        return views

def read_iphone(directory, scale, device):
    depth_file_paths = natsorted(glob.glob(os.path.join(directory, 'depth*.png')))
    color_file_paths = natsorted(glob.glob(os.path.join(directory, 'color*.png')))
    mask_file_paths = natsorted(glob.glob(os.path.join(directory, 'mask*.png')))
    
    views = []
    for image_path, depth_path, mask_path in zip(color_file_paths, depth_file_paths, mask_file_paths):
        views.append(View.load2(image_path, depth_path, mask_path, device))
    print("Found {:d} views".format(len(views)))

    if scale > 1:      
        for view in views:
            view.scale(scale)
        print("Scaled views to 1/{:d}th size".format(scale))

    return views
