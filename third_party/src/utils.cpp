#include "../include/utils.h"

int64_t neighborVoxels(const uint32_t inputCode, const int *inputIndex, int startDepth, const int depth)
{
    if (startDepth > depth)
        return -1;
    int neighborIndex[3];
    neighborIndex[0] = inputIndex[0];
    neighborIndex[1] = inputIndex[1];
    neighborIndex[2] = inputIndex[2];
    uint32_t returnCode = inputCode;
    uint32_t innerCode = (returnCode >> ((startDepth-1)*3)) & 0x07;
    int posX = innerCode & 0x01;
    int posY = (innerCode >> 1) & 0x01;
    int posZ = (innerCode >> 2) & 0x01;
    if (neighborIndex[0] != 0 && posX + neighborIndex[0] <= 1 && posX + neighborIndex[0] >= 0)   
    {
        uint32_t mask = ~(7u << (startDepth-1)*3);
        uint32_t set = (innerCode & 0x06) | (posX + neighborIndex[0]);
        returnCode = (returnCode & mask) | (set << (startDepth-1)*3);
        neighborIndex[0] = 0;
    }
    if (neighborIndex[1] != 0 && posY + neighborIndex[1] <= 1 && posY + neighborIndex[1] >= 0)   
    {
        innerCode = (returnCode >> ((startDepth-1)*3)) & 0x07;
        uint32_t mask = ~(7u << (startDepth-1)*3);
        uint32_t set = (innerCode & 0x05) | ((posY + neighborIndex[1]) << 1);
        returnCode = (returnCode & mask) | (set << (startDepth-1)*3);
        neighborIndex[1] = 0;
    }
    if (neighborIndex[2] != 0 && posZ + neighborIndex[2] <= 1 && posZ + neighborIndex[2] >= 0)   
    {
        innerCode = (returnCode >> ((startDepth-1)*3)) & 0x07;
        uint32_t mask = ~(7u << (startDepth-1)*3);
        uint32_t set = (innerCode & 0x03) | ((posZ + neighborIndex[2]) << 2);
        returnCode = (returnCode & mask) | (set << (startDepth-1)*3);
        neighborIndex[2] = 0;
    }
    if ((neighborIndex[0] != 0) || (neighborIndex[1] != 0) || ((neighborIndex[2] != 0)))
    {
        int index[3];
        index[0] = neighborIndex[0];
        index[1] = neighborIndex[1];
        index[2] = neighborIndex[2];
        int64_t neighborCode = neighborVoxels(returnCode, index, startDepth + 1, depth);
        if (neighborCode == -1)
            return -1;
        uint32_t neighborParentCode = (neighborCode >> ((startDepth-2)*3)) & 0x07;
        uint32_t mask = ~(7u << (startDepth-2)*3);
        returnCode = (neighborCode & mask) | (neighborParentCode << (startDepth-2)*3);
        uint32_t setCode = (returnCode >> ((startDepth-1)*3)) & 0x07;
        if (neighborIndex[0] == 1)  setCode &= 0x06;
        if (neighborIndex[0] == -1)  setCode |= 0x01;
        if (neighborIndex[1] == 1)  setCode &= 0x05;
        if (neighborIndex[1] == -1)  setCode |= 0x02;
        if (neighborIndex[2] == 1)  setCode &= 0x03;
        if (neighborIndex[2] == -1)  setCode |= 0x04;
        mask = ~(7u << (startDepth-1)*3);
        returnCode = (returnCode & mask) | (setCode << (startDepth-1)*3);
    }
    return returnCode;
}

static inline Eigen::Vector3d sdfInterp(const Eigen::Vector3d p1, const Eigen::Vector3d p2, float valp1, float valp2) 
{
    if (fabs(0.0f - valp1) < 1.0e-5f) return p1;
	if (fabs(0.0f - valp2) < 1.0e-5f) return p2;
	if (fabs(valp1 - valp2) < 1.0e-5f) return p1;

	float w2 = (0.0f - valp1) / (valp2 - valp1);
	float w1 = 1 - w2;

	return Eigen::Vector3d(p1[0] * w1 + p2[0] * w2,
                          p1[1] * w1 + p2[1] * w2,
                          p1[2] * w1 + p2[2] * w2);
}


void marchingCubesDense(std::vector<Eigen::Vector3d> &verts, const Eigen::Vector3i* valid_cords,
                            const Eigen::Tensor<double, 3> &dense_sdf, const std::vector<bool> mask,
                            const uint32_t num_lif, const Eigen::Vector3d offsets, const float scale)
{
    for (uint32_t i = 0; i < num_lif; ++i)
    {
        if (mask[i])
            continue;
        // Find all 8 neighbours
        Eigen::Vector3d points[8];
        float sdf_vals[8];

        sdf_vals[0] = dense_sdf(valid_cords[i][0], valid_cords[i][1], valid_cords[i][2]);
        points[0] = Eigen::Vector3d(valid_cords[i][0], valid_cords[i][1], valid_cords[i][2]);

        sdf_vals[1] = dense_sdf(valid_cords[i][0] + 1, valid_cords[i][1], valid_cords[i][2]);
        points[1] = Eigen::Vector3d(valid_cords[i][0] + 1, valid_cords[i][1], valid_cords[i][2]);

        sdf_vals[2] = dense_sdf(valid_cords[i](0) + 1, valid_cords[i](1) + 1, valid_cords[i](2));
        points[2] = Eigen::Vector3d(valid_cords[i][0] + 1, valid_cords[i][1] + 1, valid_cords[i][2]);

        sdf_vals[3] = dense_sdf(valid_cords[i][0], valid_cords[i][1] + 1, valid_cords[i][2]);
        points[3] = Eigen::Vector3d(valid_cords[i][0], valid_cords[i][1] + 1, valid_cords[i][2]);

        sdf_vals[4] = dense_sdf(valid_cords[i][0], valid_cords[i][1], valid_cords[i][2] + 1);
        points[4] = Eigen::Vector3d(valid_cords[i][0], valid_cords[i][1], valid_cords[i][2] + 1);

        sdf_vals[5] = dense_sdf(valid_cords[i][0] + 1, valid_cords[i][1], valid_cords[i][2] + 1);
        points[5] = Eigen::Vector3d(valid_cords[i][0] + 1, valid_cords[i][1], valid_cords[i][2] + 1);

        sdf_vals[6] = dense_sdf(valid_cords[i][0] + 1, valid_cords[i][1] + 1, valid_cords[i][2] + 1);
        points[6] = Eigen::Vector3d(valid_cords[i][0] + 1, valid_cords[i][1] + 1, valid_cords[i][2] + 1);
        
        sdf_vals[7] = dense_sdf(valid_cords[i][0], valid_cords[i][1] + 1, valid_cords[i][2] + 1);
        points[7] = Eigen::Vector3d(valid_cords[i][0], valid_cords[i][1] + 1, valid_cords[i][2] + 1);

        // Find triangle config.
        int cube_type = 0;
        if (sdf_vals[0] < 0) cube_type |= 1; 
        if (sdf_vals[1] < 0) cube_type |= 2;
        if (sdf_vals[2] < 0) cube_type |= 4; 
        if (sdf_vals[3] < 0) cube_type |= 8;
        if (sdf_vals[4] < 0) cube_type |= 16; 
        if (sdf_vals[5] < 0) cube_type |= 32;
        if (sdf_vals[6] < 0) cube_type |= 64; 
        if (sdf_vals[7] < 0) cube_type |= 128;

        // Find vertex position on each edge (weighted by sdf value)
        int edge_config = edgeTable[cube_type];
        Eigen::Vector3d vert_list[12];

        if (edge_config == 0) continue;
        if (edge_config & 1) vert_list[0] = sdfInterp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
        if (edge_config & 2) vert_list[1] = sdfInterp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
        if (edge_config & 4) vert_list[2] = sdfInterp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
        if (edge_config & 8) vert_list[3] = sdfInterp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
        if (edge_config & 16) vert_list[4] = sdfInterp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
        if (edge_config & 32) vert_list[5] = sdfInterp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
        if (edge_config & 64) vert_list[6] = sdfInterp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
        if (edge_config & 128) vert_list[7] = sdfInterp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
        if (edge_config & 256) vert_list[8] = sdfInterp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
        if (edge_config & 512) vert_list[9] = sdfInterp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
        if (edge_config & 1024) vert_list[10] = sdfInterp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
        if (edge_config & 2048) vert_list[11] = sdfInterp(points[3], points[7], sdf_vals[3], sdf_vals[7]);

        Eigen::Vector3d vp[3];
        // Write triangles to array.
        for (int i = 0; triangleTable[cube_type][i] != -1; i += 3) 
        {
            for (int vi = 0; vi < 3; ++vi) 
            {
                vp[vi] = vert_list[triangleTable[cube_type][i + vi]];
            }
            for (int vi = 0; vi < 3; ++ vi)
            {
                verts.push_back((vp[vi] * scale) + offsets);
            }
        }
    }
}

torch::Tensor transformToPointCloud(torch::Tensor depth, torch::Tensor K, torch::Tensor Rt)
{
    int h = depth.size(0);
    int w = depth.size(1);
    float cx = K[0][2].item<float>();
    float cy = K[1][2].item<float>();
    float fx = K[0][0].item<float>();
    float fy = K[1][1].item<float>();
    
    auto grids = torch::meshgrid({torch::arange(w, torch::kFloat32), torch::arange(h, torch::kFloat32)});
    torch::Tensor u = grids[0].t().flatten();
    torch::Tensor v = grids[1].t().flatten();

    torch::Tensor Zc = depth.view({-1});
    torch::Tensor Xc = (u - cx) * Zc / fx;
    torch::Tensor Yc = (v - cy) * Zc / fy;
    torch::Tensor Oc = torch::ones({h * w});
    torch::Tensor camera_coords = torch::stack({Xc, Yc, Zc, Oc});

    torch::Tensor world_coords = Rt.mm(camera_coords);
    torch::Tensor points = world_coords.t().slice(1, 0, 3);

    return points;
}
