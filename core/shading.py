import torch
import torch.nn as nn
from typing import Callable, Dict, List

from core.view import View
from core.sh import get_radiance

class RandomSamples(object):
    # 用于图像像素的随机采样
    def __init__(self, h, w, percentage=.5):
        self.idx = torch.randperm(h*w)[:int(h*w*percentage)]    # 将h*w个数字随机打乱后取百分之percentage的数字

    def __call__(self, tensor):
        """ Select samples from a tensor.

        Args:
            tensor: Tensor to select samples from (HxWxC or NxC)
        """
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(-1)
            out = tensor.view(-1, tensor.shape[-1])[self.idx]
            out = out.squeeze()
        else:
            out = tensor.view(-1, tensor.shape[-1])[self.idx]
        return out

def shading_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], rast, color_loss=None, shading_percentage=1):
    loss = 0
    sample_fn = lambda x: x
    for view, gbuffer in zip(views, gbuffers):
        # Get valid area
        mask = ((view.mask > 0) & (gbuffer["mask"] > 0)).squeeze()

        # Sample only within valid area
        if shading_percentage != 1:
            sample_fn = RandomSamples(view.mask[mask].shape[0], 1, shading_percentage)  # 构建随机采样器
    
        target_img = sample_fn(view.color[mask])
        target_dep = sample_fn(view.depth[mask])

        depth = sample_fn(gbuffer["depth"][mask])
        albedo = sample_fn(gbuffer["albedo"][mask])
        sh = sample_fn(gbuffer["sh"][mask])
        normal = sample_fn(gbuffer["normal"][mask])
        
        if color_loss is not None:
            depth_mask = (torch.abs(target_dep - depth) < 0.1).squeeze()
            rast = sample_fn(rast[mask])[depth_mask]
            rast_in = rast.detach().clone().to(torch.long)
            rast_in -= 1        # 减去一个偏移 https://nvlabs.github.io/nvdiffrast/

        radiance = get_radiance(sh, normal, 2).unsqueeze(-1)
        rgb = albedo * radiance
        rgb = torch.clamp(rgb, min=0.0, max=1.0)
        if color_loss is not None:
            loss_map = torch.abs(rgb[depth_mask] - target_img[depth_mask]) # torch.abs(shader(features)[depth_mask] - target_img[depth_mask])
        else:
            loss_map = torch.abs(rgb - target_img)
        loss += torch.mean(loss_map)     # shader(features)
        # debug
        if color_loss is not None:
            color_loss[rast_in, 0] = torch.mean(loss_map, dim=1)
            color_loss[rast_in, 1] += 1

    return loss / len(views)
        