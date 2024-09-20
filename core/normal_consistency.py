import torch

from core.mesh import Mesh

def normal_consistency_loss(mesh: Mesh):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """
    
    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]], mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)
    
    return (loss**2).mean()

def normal_planar_loss(mesh: Mesh):
    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh._planar], mesh._planar_normals[mesh._planar], dim=1)
    
    return (loss**2).mean()
