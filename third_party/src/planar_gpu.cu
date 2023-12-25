#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAEvent.h>
#include <cmath>
#define BLOCK 512

__global__ void find_planar_kernel(
    const torch::PackedTensorAccessor<long, 2, torch::RestrictPtrTraits, size_t> indices,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> vertexNormals,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> faceNormals,
    torch::PackedTensorAccessor<bool, 1, torch::RestrictPtrTraits, size_t> isPlanar,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> planarNormals,
    const long faceNum,
    const float thv
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < faceNum) {
        float magnitude0 = 0.0;
        float magnitude1 = 0.0;
        float magnitude2 = 0.0;
        float product0 = 0.0;
        float product1 = 0.0;
        float product2 = 0.0;
        float similarity0 = 0.0;
        float similarity1 = 0.0;
        float similarity2 = 0.0;
        // 计算模长 和 dot_product
        for (int i = 0; i < 3; i++) {
            float val0 = vertexNormals[indices[idx][0]][i];
            float val1 = vertexNormals[indices[idx][1]][i];
            float val2 = vertexNormals[indices[idx][2]][i];
            product0 += val0 * val1;
            product1 += val0 * val2;
            product2 += val1 * val2;
            magnitude0 += val0 * val0;
            magnitude1 += val1 * val1;
            magnitude2 += val2 * val2;
        }
        magnitude0 = sqrt(magnitude0);
        magnitude1 = sqrt(magnitude1);
        magnitude2 = sqrt(magnitude2);
        // 计算余弦相似度
        similarity0 = product0 / (magnitude0 * magnitude1);
        similarity1 = product1 / (magnitude0 * magnitude2);
        similarity2 = product2 / (magnitude1 * magnitude2);
        // 保存法向量
        planarNormals[idx][0] = faceNormals[idx][0];
        planarNormals[idx][1] = faceNormals[idx][1];
        planarNormals[idx][2] = faceNormals[idx][2];
        // 判断是否是平面
         if (similarity0 > thv && similarity1 > thv && similarity2 > thv)
         {
            isPlanar[idx] = true;
         }
    }
}

void find_planar_kernel_wrapper(
    const long faceNum, const float thv, 
    at::Tensor indices, 
    at::Tensor vertexNormals,
    at::Tensor faceNormals, 
    at::Tensor isPlanar, 
    at::Tensor planarNormals
){
    int grid = (faceNum + BLOCK - 1) / BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    find_planar_kernel<<<grid, BLOCK, 0, stream>>>(
        indices.packed_accessor<long, 2, torch::RestrictPtrTraits, size_t>(),
        vertexNormals.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        faceNormals.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        isPlanar.packed_accessor<bool, 1, torch::RestrictPtrTraits, size_t>(),
        planarNormals.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        faceNum,
        thv
    );
}