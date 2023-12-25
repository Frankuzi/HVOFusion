'''
Author: Frankuzi
Date: 2023-10-31 13:11:54
LastEditors: Lily 2810377865@qq.com
LastEditTime: 2023-11-22 23:07:34
FilePath: /explictRender/core/laplacian.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import torch

from core.mesh import Mesh

def laplacian_loss(mesh: Mesh):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """

    L = mesh.laplacian
    V = mesh.vertices
    
    loss = L.mm(V)
    loss[~mesh._verts_mask] = 0
    loss = loss.norm(dim=1)**2
    
    return loss.mean()