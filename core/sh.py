'''
Author: Frankuzi
Date: 2023-11-23 23:07:24
LastEditors: Lily 2810377865@qq.com
LastEditTime: 2023-11-25 15:42:49
FilePath: /explictRender/core/sh.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.size(-2)
    assert sh[0].shape[-1] == 3

    result = C0 * sh[:, 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[:, 1] +
                  C1 * z * sh[:, 2] -
                  C1 * x * sh[:, 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[:, 4] +
                      C2[1] * yz * sh[:, 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[:, 6] +
                      C2[3] * xz * sh[:, 7] +
                      C2[4] * (xx - yy) * sh[:, 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[9] +
                          C3[1] * xy * z * sh[10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[13] +
                          C3[5] * z * (xx - yy) * sh[14] +
                          C3[6] * x * (xx - 3 * yy) * sh[15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[16] +
                              C4[1] * yz * (3 * xx - yy) * sh[17] +
                              C4[2] * xy * (7 * zz - 1) * sh[18] +
                              C4[3] * yz * (7 * zz - 3) * sh[19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[20] +
                              C4[5] * xz * (7 * zz - 3) * sh[21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[24])
    return result

def get_matrix(normal, degree=3):
    if isinstance(normal, np.ndarray):
        matrix = np.zeros((normal.shape[0], degree**2))
    elif isinstance(normal, torch.Tensor):
        matrix = torch.zeros(normal.shape[0], degree**2, device=normal.device)

    matrix[:,0] = 1
    if degree > 1:
        matrix[:,1] = normal[:,1]
        matrix[:,2] = normal[:,2]
        matrix[:,3] = normal[:,0]
    if degree > 2:
        matrix[:,4] = normal[:,0] * normal[:,1]
        matrix[:,5] = normal[:,1] * normal[:,2]
        matrix[:,6] = (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        matrix[:,7] = normal[:,2] * normal[:,0]
        matrix[:,8] = (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return matrix

def get_radiance(coeff, normal, degree=3):
    '''
    coeff 9 or n 9
    normal n 3
    '''

    radiance = coeff[...,0]
    if degree > 1:
        radiance = radiance + coeff[...,1] * normal[:,1]
        radiance = radiance + coeff[...,2] * normal[:,2]
        radiance = radiance + coeff[...,3] * normal[:,0]
    if degree > 2:
        radiance = radiance + coeff[...,4] * normal[:,0] * normal[:,1]
        radiance = radiance + coeff[...,5] * normal[:,1] * normal[:,2]
        radiance = radiance + coeff[...,6] * (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        radiance = radiance + coeff[...,7] * normal[:,2] * normal[:,0]
        radiance = radiance + coeff[...,8] * (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return radiance