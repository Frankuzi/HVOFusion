from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from core.view_sampler import ViewSampler
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import nvdiffrast.torch as dr

from core.config import load_config
from core.io import read_views, read_iphone, read_mesh, write_mesh
from core.view_sampler import ViewSampler
from core.geometry import AABB
from core.renderer import Renderer
from core.shading import shading_loss
from core.mask import mask_loss
from core.normal_consistency import normal_consistency_loss
from core.laplacian import laplacian_loss
from core.mesh import Mesh, sample_surface
from core.chamfer import chamfer_distance
from core.sh import get_matrix, get_radiance

torch.classes.load_library(
    "third_party/build/lib.linux-x86_64-3.8/svo.cpython-38-x86_64-linux-gnu.so")


def main():    
    ### Step1: 读取配置文件
    parser = ArgumentParser(description='Test', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    ### Step2: 构建输出文件夹
    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and cfg['device'] >= 0:
        device = torch.device(f"cuda:{cfg['device']}")
    print(f"Using device {device}")
    # Create directories
    input_dir = Path(cfg['input_dir'])
    output_dir = Path(cfg['output_dir'])
    run_name = cfg['run_name'] if cfg['run_name'] is not None else input_dir.parent.name
    experiment_dir = output_dir / run_name
    process_images_save_path = experiment_dir / "process_images"
    process_meshes_save_path = experiment_dir / "process_meshes"
    process_images_save_path.mkdir(parents=True, exist_ok=True)
    process_meshes_save_path.mkdir(parents=True, exist_ok=True)
    results_images_save_path = experiment_dir / "results_images"
    results_meshes_save_path = experiment_dir / "results_meshes"
    results_images_save_path.mkdir(parents=True, exist_ok=True)
    results_meshes_save_path.mkdir(parents=True, exist_ok=True)

    ### Step3: 读取数据集中的相机图像等信息 读取bound信息
    views = read_views(input_dir, cfg['datasets_name'], device=device)
    # Configure the view sampler
    view_sampler = ViewSampler(views=views, **ViewSampler.get_parameters(cfg))
    # Read AABB Bound
    aabb = AABB(np.array([[cfg['bound'][0][0], cfg['bound'][1][0], cfg['bound'][2][0]], [cfg['bound'][0][1], cfg['bound'][1][1], cfg['bound'][2][1]]], dtype=np.float32))
    
    ### Step4: 初始化SVO 初始化用于保存mesh顶点和index的变量
    svo = torch.classes.svo.Octree()
    svo.init(torch.tensor(aabb.center), aabb.longest_extent, cfg['minExtent'], cfg['minSize'], cfg['pointsValid'], cfg['normalRadius'], cfg['curvatureTHR'], \
        cfg['sdfRadius'], cfg['reconTHR'], cfg['minBorder'], cfg['render_interval'], cfg['subLevel'], cfg['weightMode'], cfg['allSampleMode'])
    
    ### Step5: 创建Render 颜色优化变量和NeuralShader
    renderer = Renderer(device=device)
    renderer.set_near_far(views, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)    # 根据相机的观测视角设置最近和最远渲染平面
    
    ### Step6: Loss函数和权重初始化
    loss_weights = {
        "mask": cfg['weight_mask'],
        "normal": cfg['weight_normal_consistency'],
        "laplacian": cfg['weight_laplacian'],
        "shading": cfg['weight_shading'],
        "depth": cfg['weight_depth'],
        "edge": cfg['weight_edge'],
    }
    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}    # 创建字典保存loss值

    for i in tqdm(range(len(views))):
        svo.updateTree(views[i].depth.squeeze().cpu(), views[i].camera.K.cpu(), views[i].camera.Rt.cpu(), i)
        if (i+1) % cfg['render_interval'] == 0:
            # 获得mc结果 获得点云结果 创建mesh
            verts, faces, index, pcd_points, pcd_normals, verts_mask, faces_mask = svo.packOutput()       
            pcd_points = pcd_points.unsqueeze(0).to(device)
            pcd_normals = pcd_normals.unsqueeze(0).to(device)
            mesh_initial = Mesh(verts, faces, None, device=device)      # 构建mesh
            mesh_initial.compute_connectivity()
            bound = np.stack([aabb.minmax[0], aabb.minmax[1]]).transpose(1, 0)
            bound = torch.from_numpy(bound).to(device)
            # debug
            # tmp_mesh = o3d.geometry.TriangleMesh()
            # tmp_mesh.vertices = o3d.utility.Vector3dVector(mesh_initial.vertices.detach().cpu().numpy())
            # tmp_mesh.triangles = o3d.utility.Vector3iVector(mesh_initial.indices.detach().cpu().numpy())
            # o3d.io.write_triangle_mesh("test.ply", tmp_mesh)
            # 创建顶点优化变量
            lr_vertices = cfg['lr_vertices']
            vertex_offsets = torch.zeros_like(mesh_initial.vertices)    # [xxx, 3] 创建一个zero数组用于优化顶点偏移量
            vertex_offsets.requires_grad = True
            optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)
            # 创建albedo优化变量
            vertex_albedo = torch.ones_like(mesh_initial.vertices) * 0.01
            vertex_albedo.requires_grad = True
            # 创建spherical harmonics coefficients优化变量 
            # compute sphere harmonic coefficient as initialization 预计算球谐系数
            with torch.no_grad():
                valid_normals = []
                valid_grayimgs = []
                for k in range(len(views)):        # i+1-cfg['render_interval'], i+1
                    P = Renderer.to_gl_camera(views[k].camera, views[k].resolution, n=renderer.near, f=renderer.far)
                    pos_clip = Renderer.transform_pos(P, mesh_initial.vertices)
                    idx = mesh_initial.indices.int()
                    rast, _ = dr.rasterize(renderer.glctx, pos_clip, idx, resolution=views[k].resolution)
                    normal, _ = dr.interpolate(mesh_initial.vertex_normals[None, ...], rast, idx)
                    normal = dr.antialias(normal, rast, pos_clip, idx)
                    valid_idx = ((views[k].mask > 0) & (torch.clamp(rast[..., -1:], 0, 1) > 0)).squeeze(-1)
                    valid_normals.append(normal[valid_idx].detach().cpu().numpy())
                    valid_grayimgs.append(views[k].gray.unsqueeze(0)[valid_idx].detach().cpu().numpy())
            valid_normals = np.concatenate(valid_normals, axis=0)
            valid_grayimgs = np.concatenate(valid_grayimgs, axis=0)
            matrix = get_matrix(valid_normals, 2)
            sh_coeff = np.linalg.lstsq(matrix, valid_grayimgs, rcond=None)[0]
            sh_coeff = np.tile(sh_coeff, (mesh_initial.vertices.size(0), 1))
            vertex_sh = torch.from_numpy(sh_coeff.astype(np.float32)).to(device)
            vertex_sh.requires_grad_(True)
            optimizer_colors = torch.optim.Adam([{'params': vertex_albedo, 'lr': cfg['lr_color']}, {'params': vertex_sh, 'lr': cfg['lr_sh']}])
            # 创建颜色损失列表
            color_losses = []
            color_loss = torch.zeros([mesh_initial.indices.size(0), 2], device=device)
            
            # 主循环
            iterations = cfg['iterations']
            progress_bar = tqdm(range(1, iterations + 1), leave=False)
            for iteration in progress_bar:
                progress_bar.set_description(desc=f'Iteration {iteration}')
                # Deform the initial mesh
                mesh = mesh_initial.with_vertices(mesh_initial.vertices + vertex_offsets)   # 每次训练都会把顶点偏移量更新一下
                recon_xyz, recon_normals = sample_surface(mesh.indices[faces_mask], mesh.vertices.unsqueeze(0), int(pcd_points.size(1)*1.5))
                views_subset, views_index = view_sampler(views[i+1-cfg['render_interval']: i+1])     # views[i+1-cfg['render_interval']: i+1]
                # pcd1 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(recon_xyz.squeeze().detach().cpu().numpy())
                # # pcd1.normals = o3d.utility.Vector3dVector(recon_normals.squeeze().detach().cpu().numpy())
                # o3d.io.write_point_cloud("pcd1.ply", pcd1)
                # pcd2 = o3d.geometry.PointCloud()
                # pcd2.points = o3d.utility.Vector3dVector(pcd_points.squeeze().detach().cpu().numpy())
                # # pcd2.normals = o3d.utility.Vector3dVector(pcd_normals.squeeze().detach().cpu().numpy())
                # o3d.io.write_point_cloud("pcd2.ply", pcd2)
                fbuffers, rast = renderer.render_color(views_subset, mesh, vertex_albedo, vertex_sh, with_antialiasing=False)
                if loss_weights['mask'] > 0:
                    losses['mask'] = mask_loss(views_subset, fbuffers)
                if loss_weights['normal'] > 0:
                    losses['normal'] = normal_consistency_loss(mesh)
                if loss_weights['laplacian'] > 0:
                    losses['laplacian'] = laplacian_loss(mesh)
                if loss_weights['shading'] > 0:
                    losses['shading'] = shading_loss(views_subset, fbuffers, rast, color_loss, shading_percentage=cfg['shading_percentage'])
                if loss_weights['depth'] > 0:
                    xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, pcd_points, x_normals=recon_normals, y_normals=pcd_normals, unoriented=True)
                    losses['depth'] = xyz_chamfer_loss # + 0.1 * normals_chamfer_loss
                if loss_weights['edge'] > 0:
                    a = mesh.vertices[mesh.indices[:, 0].long()]
                    b = mesh.vertices[mesh.indices[:, 1].long()]
                    c = mesh.vertices[mesh.indices[:, 2].long()]
                    losses['edge'] =  torch.cat([((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)]).mean()
                loss = torch.tensor(0., device=device)
                for k, v in losses.items():
                    loss += v * loss_weights[k]
                optimizer_vertices.zero_grad()
                optimizer_colors.zero_grad()
                loss.backward()
                vertex_offsets.grad[~verts_mask] = 0
                optimizer_vertices.step()
                optimizer_colors.step()
                progress_bar.set_postfix({'depth loss': losses['depth'].detach().cpu(), 'shading loss': losses['shading'].detach().cpu()})
                
                # 更新颜色loss列表
                if cfg['view_sampling_mode'] == 'importance_sampling':
                    if len(color_losses) < cfg['render_interval']:
                        color_losses.append(losses['shading'].detach().cpu().numpy())
                    else:
                        color_losses[views_index] = losses['shading'].detach().cpu().numpy()
                    if len(color_losses) == cfg['render_interval']:
                        view_sampler.update_probabilities(color_losses)
                
                if (iteration % cfg['iterations'] == 0):
                    with torch.no_grad():
                        debug_view, _ = view_sampler(views[i+1-cfg['render_interval']: i+1])
                        debug_view = debug_view[0]
                        debug_gbuffer, _ = renderer.render_color([debug_view], mesh, vertex_albedo, vertex_sh, with_antialiasing=False)
                        debug_gbuffer = debug_gbuffer[0]
                        normals = debug_gbuffer["normal"]
                        albedo = debug_gbuffer["albedo"]
                        sh = debug_gbuffer["sh"]
                        normals = normals.view(-1, normals.size(-1))
                        albedo = albedo.view(-1, albedo.size(-1))
                        sh = sh.view(-1, sh.size(-1))
                        # Save the depth rendering
                        debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'depth'], with_antialiasing=False)[0]
                        depth_predict = debug_gbuffer['depth'].squeeze(-1).detach().cpu().numpy()
                        depth = debug_view.depth.squeeze(-1).cpu().numpy()
                        shaded_path = (process_images_save_path / "depth")
                        shaded_path.mkdir(parents=True, exist_ok=True)
                        plt.imsave(shaded_path / f'depthshading_{i}_{iteration:06d}.png', depth_predict, cmap='gray', vmax=5)
                        plt.imsave(shaded_path / f'depthtruth_{i}_{iteration:06d}.png', depth, cmap='gray', vmax=5)
                        # Save the shaded rendering
                        radiance = get_radiance(sh, normals, 2).unsqueeze(-1)
                        rgb = albedo * radiance
                        rgb = rgb.view(debug_view.resolution[0], debug_view.resolution[1], -1)
                        shaded_image = rgb * debug_view.mask + (1-debug_view.mask) # rgb.view(debug_view.resolution[0], debug_view.resolution[1], -1) * debug_view.mask + (1-debug_view.mask) # shader(features) * debug_view.mask + (1-debug_view.mask)
                        shaded_path = (process_images_save_path / "shaded")
                        shaded_path.mkdir(parents=True, exist_ok=True)
                        shaded_image = torch.clamp(shaded_image, min=0.0, max=1.0)
                        plt.imsave(shaded_path / f'neuralshading_{i}_{iteration:06d}.png', shaded_image.detach().cpu().numpy())
                        # Save the mesh color
                        radiance = get_radiance(vertex_sh, mesh.vertex_normals, 2).unsqueeze(-1)
                        vertex_colors = vertex_albedo * radiance
                        vertex_colors = torch.clamp(vertex_colors, min=0.0, max=1.0)
                        mesh.colors = vertex_colors
                        mesh_for_writing = mesh.detach().to('cpu')
                        write_mesh(process_meshes_save_path / f"mesh_{i}_{iteration:06d}.ply", mesh_for_writing)
                        # debug
                        loss_map = color_loss[:, 0] / color_loss[:, 1]
                        loss_map[torch.isnan(loss_map)] = -1
                        index_mask = torch.ones_like(loss_map, dtype=torch.bool)
                        index_mask[loss_map == -1] = False
                        color_mask = torch.zeros([mesh.vertices.size(0)], dtype=torch.bool)
                        color_mask[torch.unique(mesh.indices[index_mask])] = True
                        svo.updateVerts(vertex_offsets.detach().cpu(), vertex_colors.detach().cpu(), vertex_albedo.detach().cpu(), 
                                        vertex_sh.detach().cpu(), index.unsqueeze(-1).detach().cpu(), color_mask.detach().cpu())
    
    print('Packout...')
    verts, faces, color, albedo, sh = svo.packVerts()
        
    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(verts)
    mesh_out.triangles = o3d.utility.Vector3iVector(faces)
    mesh_out.vertex_colors  = o3d.utility.Vector3dVector(color)
    
    # 是否进行后处理
    if cfg['postprocess']:
        print('Postprocessing...')
        # mesh_out = mesh_out.filter_smooth_laplacian(number_of_iterations=1)
        mesh = Mesh(mesh_out.vertices, mesh_out.triangles, None, device=device)
        mesh.compute_connectivity()
        bound = np.stack([aabb.minmax[0], aabb.minmax[1]]).transpose(1, 0)
        bound = torch.from_numpy(bound).to(device)
        view_sampler = ViewSampler(views=views, mode='random', views_per_iter=1)
        vertex_albedo = albedo.clone().to(torch.float32).to(device)
        vertex_albedo.requires_grad = True
        vertex_sh = sh.clone().to(torch.float32).to(device)
        vertex_sh.requires_grad = True
        optimizer_colors = torch.optim.Adam([{'params': vertex_albedo, 'lr': 0.0001}, {'params': vertex_sh, 'lr': 0.0001}])
        # scheduler_colors = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_colors, T_max=cfg['iterations_after'], eta_min=0.000001)
        
        iterations = cfg['iterations_after']
        progress_bar = tqdm(range(1, iterations + 1))
        for iteration in progress_bar:
            progress_bar.set_description(desc=f'Iteration {iteration}')
            views_subset, _ = view_sampler(views)
            fbuffers, rast = renderer.render_color(views_subset, mesh, vertex_albedo, vertex_sh, with_antialiasing=False)
            loss = shading_loss(views_subset, fbuffers, rast)
            optimizer_colors.zero_grad()
            loss.backward()
            if torch.isnan(vertex_albedo.grad).any():
                vertex_albedo.grad[torch.isnan(vertex_albedo.grad)] = 0
            if torch.isnan(vertex_sh.grad).any():
                vertex_sh.grad[torch.isnan(vertex_sh.grad)] = 0
            optimizer_colors.step()
            # scheduler_colors.step()
            progress_bar.set_postfix({'shading loss': loss.detach().cpu()})
        radiance = get_radiance(vertex_sh, mesh.vertex_normals, 2).unsqueeze(-1)
        vertex_colors = vertex_albedo * radiance
        vertex_colors = torch.clamp(vertex_colors, min=0.0, max=1.0)
        mesh_out.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.detach().cpu())
    
    # 渲染图像 保存mesh
    render_index = [10]
    for i in render_index:
        with torch.no_grad():
            debug_view = views[i]
            debug_gbuffer, _ = renderer.render_color([debug_view], mesh, vertex_albedo, vertex_sh, with_antialiasing=False)
            debug_gbuffer = debug_gbuffer[0]
            normals = debug_gbuffer["normal"]
            albedo = debug_gbuffer["albedo"]
            sh = debug_gbuffer["sh"]
            normals = normals.view(-1, normals.size(-1))
            albedo = albedo.view(-1, albedo.size(-1))
            sh = sh.view(-1, sh.size(-1))
            # Save the depth rendering
            debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'depth'], with_antialiasing=False)[0]
            depth_predict = debug_gbuffer['depth'].squeeze(-1).detach().cpu().numpy()
            depth = debug_view.depth.squeeze(-1).cpu().numpy()
            shaded_path = (results_images_save_path / "depth")
            shaded_path.mkdir(parents=True, exist_ok=True)
            plt.imsave(shaded_path / f'depthshading_{i}_{iteration:06d}.png', depth_predict, cmap='gray', vmax=5)
            plt.imsave(shaded_path / f'depthtruth_{i}_{iteration:06d}.png', depth, cmap='gray', vmax=5)
            # Save the shaded rendering
            radiance = get_radiance(sh, normals, 2).unsqueeze(-1)
            rgb = albedo * radiance
            rgb = rgb.view(debug_view.resolution[0], debug_view.resolution[1], -1)
            shaded_image = rgb * debug_view.mask + (1-debug_view.mask) # rgb.view(debug_view.resolution[0], debug_view.resolution[1], -1) * debug_view.mask + (1-debug_view.mask) # shader(features) * debug_view.mask + (1-debug_view.mask)
            shaded_path = (results_images_save_path / "shaded")
            shaded_path.mkdir(parents=True, exist_ok=True)
            shaded_image = torch.clamp(shaded_image, min=0.0, max=1.0)
            plt.imsave(shaded_path / f'neuralshading_{i}_{iteration:06d}.png', shaded_image.detach().cpu().numpy())
    o3d.io.write_triangle_mesh(os.path.join(results_meshes_save_path, f"mesh_increase_{i}_{iteration:06d}.ply"), mesh_out)
    print(results_meshes_save_path / f"mesh_increase_{i}_{iteration:06d}.ply saved!")

if __name__ == '__main__':
    main()