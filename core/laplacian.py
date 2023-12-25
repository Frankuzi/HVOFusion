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