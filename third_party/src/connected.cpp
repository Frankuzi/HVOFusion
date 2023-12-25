/*
 * @Author: Frankuzi
 * @Date: 2023-11-06 17:15:20
 * @LastEditors: Lily 2810377865@qq.com
 * @LastEditTime: 2023-11-22 17:53:25
 * @FilePath: /explictRender/third_party/src/connected.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#include <torch/extension.h>

torch::Tensor computeFaceCorrespondences(const torch::Tensor indicesTensor, int countsMax, int countsSize, int indicesSize)
{
    auto inverseIndices = indicesTensor.accessor<int64_t, 1>();

    torch::Tensor face_ids = torch::arange(indicesSize);
    face_ids = face_ids.unsqueeze(-1).repeat({1, 3}).flatten();

    torch::Tensor face_correspondences = torch::zeros({countsSize, countsMax}, torch::kLong);
    torch::Tensor face_correspondences_indices = torch::zeros({countsSize}, torch::kLong);

    for (int64_t ei = 0; ei < inverseIndices.size(0); ei++) 
    {
        int64_t ei_unique = inverseIndices[ei];
        face_correspondences[ei_unique][face_correspondences_indices[ei_unique]] = face_ids[ei];
        face_correspondences_indices[ei_unique] += 1;
    }

    return face_correspondences;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("computeFaceCorrespondences", &computeFaceCorrespondences, "computeFaceCorrespondences");
}