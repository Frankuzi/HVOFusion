import math
import torch

# 球谐系数相关参数
HALF_SQRT_3_BY_PI = 0.5 * math.sqrt(3. / math.pi)
HALF_SQRT_15_BY_PI = 0.5 * math.sqrt(15. / math.pi)
QUARTER_SQRT_15_BY_PI = 0.25 * math.sqrt(15. / math.pi)
QUARTER_SQRT_5_BY_PI = 0.25 * math.sqrt(5. / math.pi)

Y_0_0 = 0.5 * math.sqrt(1. / math.pi)
Y_m1_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * torch.sin(theta) * torch.sin(phi)
Y_0_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * torch.cos(theta)
Y_1_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * torch.sin(theta) * torch.cos(phi)
Y_m2_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * torch.sin(theta) * torch.cos(phi) * torch.sin(
    theta) * torch.sin(phi)
Y_m1_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * torch.sin(theta) * torch.sin(phi) * torch.cos(theta)
Y_0_2 = lambda theta, phi: QUARTER_SQRT_5_BY_PI * (3 * torch.cos(theta) * torch.cos(theta) - 1)
Y_1_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * torch.sin(theta) * torch.cos(phi) * torch.cos(theta)
Y_2_2 = lambda theta, phi: QUARTER_SQRT_15_BY_PI * (
        torch.pow(torch.sin(theta) * torch.cos(phi), 2) - torch.pow(torch.sin(theta) * torch.sin(phi), 2))

# 计算球谐系数公式 该函数是C_0_0、C_m1_1、C_0_1、C_1_1、C_m2_2、C_m1_2、C_0_2、C_1_2、C_2_2参数化的球谐函数的线性组合
# Y_0_0是球谐函数的一个分量，它在球面上的取值是常数。Y_m1_1、Y_0_1、Y_1_1是球谐函数的三个分量，它们在球面上的取值与角度θ和φ的三角函数有
# Y_m2_2、Y_m1_2、Y_0_2、Y_1_2、Y_2_2是球谐函数的另外五个分量，它们在球面上的取值与θ和φ的三角函数的乘积有关
# 这些球谐函数的线性组合被用来表示某些函数在球面上的分布，例如用于光照计算中的环境光照、天空渲染、物体表面的反射
# 系数C_0_0、C_m1_1、C_0_1、C_1_1、C_m2_2、C_m1_2、C_0_2、C_1_2、C_2_2决定了这个线性组合的权重，从而控制了球谐函数的形状和分布
def harmonic(C_0_0, C_m1_1, C_0_1, C_1_1, C_m2_2, C_m1_2, C_0_2, C_1_2, C_2_2, theta, phi):
    return C_0_0 * (0.5 * math.sqrt(1. / math.pi)) + C_m1_1 * (HALF_SQRT_3_BY_PI * torch.sin(theta) * torch.sin(phi)) + C_0_1 * (HALF_SQRT_3_BY_PI * torch.cos(theta)) + \
    C_1_1 * (HALF_SQRT_3_BY_PI * torch.sin(theta) * torch.cos(phi)) + C_m2_2 * (HALF_SQRT_15_BY_PI * torch.sin(theta) * torch.cos(phi) * torch.sin(theta) * torch.sin(phi)) + \
    C_m1_2 * (HALF_SQRT_15_BY_PI * torch.sin(theta) * torch.sin(phi) * torch.cos(theta)) + C_0_2 * (QUARTER_SQRT_5_BY_PI * (3 * torch.cos(theta) * torch.cos(theta) - 1)) + \
    C_1_2 * (HALF_SQRT_15_BY_PI * torch.sin(theta) * torch.cos(phi) * torch.cos(theta)) + C_2_2 * QUARTER_SQRT_15_BY_PI * (torch.pow(torch.sin(theta) * torch.cos(phi), 2) - torch.pow(torch.sin(theta) * torch.sin(phi), 2))

def rgb_harmonics(rgb_harmonic_coefficients, theta, phi):
    rC_0_0 = rgb_harmonic_coefficients[:, 0]
    rC_m1_1 = rgb_harmonic_coefficients[:, 1]
    rC_0_1 = rgb_harmonic_coefficients[:, 2]
    rC_1_1 = rgb_harmonic_coefficients[:, 3]
    rC_m2_2 = rgb_harmonic_coefficients[:, 4]
    rC_m1_2 = rgb_harmonic_coefficients[:, 5]
    rC_0_2 = rgb_harmonic_coefficients[:, 6]
    rC_1_2 = rgb_harmonic_coefficients[:, 7]
    rC_2_2 = rgb_harmonic_coefficients[:, 8]
    
    gC_0_0 = rgb_harmonic_coefficients[:, 9]
    gC_m1_1 = rgb_harmonic_coefficients[:, 10]
    gC_0_1 = rgb_harmonic_coefficients[:, 11]
    gC_1_1 = rgb_harmonic_coefficients[:, 12]
    gC_m2_2 = rgb_harmonic_coefficients[:, 13]
    gC_m1_2 = rgb_harmonic_coefficients[:, 14]
    gC_0_2 = rgb_harmonic_coefficients[:, 15]
    gC_1_2 = rgb_harmonic_coefficients[:, 16]
    gC_2_2 = rgb_harmonic_coefficients[:, 17]
    
    bC_0_0 = rgb_harmonic_coefficients[:, 18]
    bC_m1_1 = rgb_harmonic_coefficients[:, 19]
    bC_0_1 = rgb_harmonic_coefficients[:, 20]
    bC_1_1 = rgb_harmonic_coefficients[:, 21]
    bC_m2_2 = rgb_harmonic_coefficients[:, 22]
    bC_m1_2 = rgb_harmonic_coefficients[:, 23]
    bC_0_2 = rgb_harmonic_coefficients[:, 24]
    bC_1_2 = rgb_harmonic_coefficients[:, 25]
    bC_2_2 = rgb_harmonic_coefficients[:, 26]
    
    red = harmonic(rC_0_0, rC_m1_1, rC_0_1, rC_1_1, rC_m2_2, rC_m1_2, rC_0_2, rC_1_2, rC_2_2, theta, phi)
    green = harmonic(gC_0_0, gC_m1_1, gC_0_1, gC_1_1, gC_m2_2, gC_m1_2, gC_0_2, gC_1_2, gC_2_2, theta, phi)
    blue = harmonic(bC_0_0, bC_m1_1, bC_0_1, bC_1_1, bC_m2_2, bC_m1_2, bC_0_2, bC_1_2, bC_2_2, theta, phi)
    
    return torch.stack([red, green, blue], dim=1)