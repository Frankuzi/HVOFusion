'''
Author: Frankuzi
Date: 2023-10-31 13:11:06
LastEditors: Lily 2810377865@qq.com
LastEditTime: 2023-11-22 18:02:00
FilePath: /explictRender/core/normal_consistency.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import torch

from core.mesh import Mesh

def normal_consistency_loss(mesh: Mesh):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """
    # 这里认为相邻的两个面的法向量是一致的
    
    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]], mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)
    
    return (loss**2).mean()

def normal_planar_loss(mesh: Mesh):
    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh._planar], mesh._planar_normals[mesh._planar], dim=1)
    
    return (loss**2).mean()