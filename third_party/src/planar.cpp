#include <torch/extension.h>
#include "../include/planar.h"

std::tuple<at::Tensor, at::Tensor> findPlanar(const at::Tensor &indices, const at::Tensor &vertexNormals, const at::Tensor &faceNormals, float thv) 
{
    CHECK_CONTIGUOUS(indices);
    CHECK_CONTIGUOUS(vertexNormals);
    CHECK_CONTIGUOUS(faceNormals);
    CHECK_IS_LONG(indices);
    CHECK_IS_FLOAT(vertexNormals);
    CHECK_IS_FLOAT(faceNormals);

    at::Tensor isPlanar = torch::zeros({indices.size(0)}, at::device(indices.device()).dtype(at::ScalarType::Bool));
    at::Tensor planarNormals = torch::zeros({indices.size(0), 3}, at::device(indices.device()).dtype(at::ScalarType::Float));
    find_planar_kernel_wrapper(indices.size(0), thv, indices, vertexNormals, faceNormals, isPlanar, planarNormals);
    
    return std::make_tuple(isPlanar, planarNormals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("findPlanar", &findPlanar, "findPlanar");
}