#include <omp.h>
#include "../include/octree.h"
#include "../include/normals.hpp"
#include "../include/utils.h"

/******************************** Octree 相关函数 **********************************/
/**
 * @description: Octree 构造函数(默认)
 * @return {*}
 */
Octree::Octree()
    : root_(0), extent_(0), data_(0), normal_(0), normalConf_(0), curvature_(0), verts_(0), voxel_(0), mcVoxels_(0)
{
}

/**
 * @description: Octree 构造函数(输入中心点坐标和边长)
 * @param {Tensor} center 八叉树的中心点 世界坐标系
 * @param {double} extent 空间最大边长
 * @return {*}
 */
Octree::Octree(const Eigen::Vector3d center, double extent, OctreeParams params)
{
    params_ = params;
    center_[0] = center[0];
    center_[1] = center[1];
    center_[2] = center[2];
    // 初始化root_节点
    root_ = new Octant;
    root_->x = center[0];
    root_->y = center[1];
    root_->z = center[2];
    root_->extent = extent;    // 节点边长
    root_->depth = 0;           // root节点深度为0
    extent_ = extent;
    data_ = new std::vector<Eigen::Vector3d>;
    normal_ = new std::vector<Eigen::Vector3d>;
    normalConf_ = new std::vector<bool>;
    curvature_ = new std::vector<double>;
    verts_ = new VertsArray;
    voxel_ = new std::vector<Octant*>;
    mcVoxels_ = new std::unordered_set<Octant*>;
}

/**
 * @description: Octree 析构函数
 * @return {*}
 */
Octree::~Octree()
{
    root_ = 0;
    data_ = 0;
    normal_ = 0;
    normalConf_ = 0;
    verts_ = 0;
    voxel_ = 0;
    mcVoxels_ = 0;
    delete root_;
    delete data_;
    delete normal_;
    delete normalConf_;
    delete verts_;
    delete voxel_;
    delete mcVoxels_;
}

/**
 * @description: Octree 初始化函数配合Octree()使用
 * @param {Tensor} center 八叉树的中心点 世界坐标系
 * @param {double} extent 空间最大边长
 * @return {*}
 */
void Octree::init(const torch::Tensor center, double extent, double minExtent, double minSize, int64_t pointsValid, double normalRadius,
        double curvatureTHR, double sdfRadius, int64_t reconTHR, double minBorder, int64_t mcInterval, int64_t subLevel, bool weightMode, bool allSampleMode)
{
    // 更新配置
    params_.minExtent = minExtent;
    params_.minSize = minSize;
    params_.pointsValid = pointsValid;
    params_.normalRadius = normalRadius;
    params_.curvatureTHR = curvatureTHR;
    params_.sdfRadius = sdfRadius;
    params_.reconTHR = reconTHR;
    params_.minBorder = minBorder;
    params_.mcInterval = mcInterval;
    params_.subLevel = subLevel;
    params_.weightMode = weightMode;
    params_.allSampleMode = allSampleMode;
    std::cout << "Octree params: " << std::endl;
    std::cout << "minExtent: " << params_.minExtent << std::endl;
    std::cout << "minSize: " << params_.minSize << std::endl;
    std::cout << "pointsValid: " << params_.pointsValid << std::endl;
    std::cout << "normalRadius: " << params_.normalRadius << std::endl;
    std::cout << "curvatureTHR: " << params_.curvatureTHR << std::endl;
    std::cout << "sdfRadius: " << params_.sdfRadius << std::endl;
    std::cout << "reconTHR: " << params_.reconTHR << std::endl;
    std::cout << "minBorder: " << params_.minBorder << std::endl;
    std::cout << "mcInterval: " << params_.mcInterval << std::endl;
    std::cout << "subLevel: " << params_.subLevel << std::endl;
    std::cout << "weightMode: " << params_.weightMode << std::endl;
    std::cout << "allSampleMode: " << params_.allSampleMode << std::endl;
    // 定义tensor坐标访问器
    auto points = center.accessor<float, 1>();          // 定义一个Accessors用于高效的访问points元素 二维
    if (points.size(0) != 3)                            // 确定points的第二维是3维的表示空间点
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(0) << " expect 3" << std::endl;
        return;
    }

    center_[0] = points[0];
    center_[1] = points[1];
    center_[2] = points[2];
    // 初始化root_节点
    root_ = new Octant;
    root_->x = points[0];
    root_->y = points[1];
    root_->z = points[2];
    root_->extent = extent;    // 节点边长
    root_->depth = 0;           // root节点深度为0
    extent_ = extent;
    data_ = new std::vector<Eigen::Vector3d>;
    normal_ = new std::vector<Eigen::Vector3d>;
    normalConf_ = new std::vector<bool>;
    curvature_ = new std::vector<double>;
    verts_ = new VertsArray;
    voxel_ = new std::vector<Octant*>;
    mcVoxels_ = new std::unordered_set<Octant*>;
}

/**
 * @description: 清空 Octree
 * @return {*}
 */
void Octree::clear()
{
    root_ = 0;
    extent_ = 0;
    data_ = 0;
    normal_ = 0;
    normalConf_ = 0;
    verts_ = 0;
    voxel_ = 0;
}

/**
 * @description: Octant 构造函数
 * @return {*}
 */
Octree::Octant::Octant()
    : isLeaf(true), isFixed(0), isFixedLast(0), isUpdated(false), x(0.0f), y(0.0f), z(0.0f), extent(0.0f), depth(0),
    curvature(0.0f), sdf(0), triangles(0), mortonCode(0), size(0), frames(0), weight(0), lastWeight(0), curveWeight(0)
{
    isFixed = new bool[8];      // 初始化8个顶点都可以更新
    isFixedLast = new bool[8];
    frames = new int[8];        // 初始化8个顶点frames值为-1
    sdf = new double[8];        // 初始化8个顶点sdf值为0
    weight = new uint32_t[8];     // 初始化8个顶点weight值为0
    lastWeight = new uint32_t[8];     // 初始化8个顶点lastWeight值为0
    std::fill(isFixed, isFixed + 8, false);
    std::fill(isFixedLast, isFixedLast + 8, false);
    std::fill(frames, frames + 8, -1);
    std::fill(sdf, sdf + 8, 0);
    std::fill(weight, weight + 8, 0);
    std::fill(lastWeight, lastWeight + 8, 0);
    memset(&child, 0, 8 * sizeof(Octant *));
    memset(&neighbor, 0, 26 * sizeof(Octant *));
}
 
/**
 * @description: Octant 析构函数
 * @return {*}
 */
Octree::Octant::~Octant()
{
    delete sdf;
    delete weight;
    delete triangles;
    delete frames;
    for (uint32_t i = 0; i < 8; ++i)
        delete child[i];
    for (uint32_t i = 0; i < 26; ++i)
        delete neighbor[i];
}

/**
 * @description: 向Octree中插入一帧点云并计算法向量更新SDF
 * @param {Matrix<double, Eigen::Dynamic, 3>} points 每一帧输入的点云 (n, 3)
 * @param {Vector3d} camera 相机位置 (3)
 * @return {*}
 */
void Octree::insert(const Eigen::Matrix<double, Eigen::Dynamic, 3> &points, const Eigen::Vector3d &camera)
{
    // 插入点
    std::unordered_set<Octant*> voxels;     // 记录了被插入点的voxels
    for (int i = 0; i < points.rows(); ++i)
    {
        insertOctant(points.row(i).transpose(), root_, extent_, 0, voxels);
    }
    // 计算当前帧法向量
    normal_->resize(data_->size());     // 将法向量的数组大小扩展到与data_一致
    normalConf_->resize(data_->size()); // 将法向量置信度数组大小扩展到与data_一致
    curvature_->resize(data_->size());  // 将曲率数组大小扩展到与data_一致
    updateNormals(camera, voxels);
    // 计算当前帧SDF
    updateVoxel(voxels);
}

/**
 * @description: 向Octree中插入一个点 递归插入
 * @param {Vector3d&} point 插入点坐标
 * @param {Octant*} octant  当前层级节点指针
 * @param {double} extent    当前层级节点长度
 * @param {uint32_t} morton 累加的morton码
 * @param {std::unordered_set<Octant*>} voxels 因为插入点后受到影响的voxels
 * @return {void} 
 */
void Octree::insertOctant(const Eigen::Vector3d& point, Octant* octant, double extent, uint32_t morton, std::unordered_set<Octant*> &voxels)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    
    if (extent > params_.minExtent)
    {
        if (octant->isLeaf)             
            octant->isLeaf = false;     // 节点分裂 设置为False
        
        uint32_t mortonCode = 0;
        Octant* childOctant;    
        double childExtent;
        // 每个点point和x点比较确定在x周围8个区域的哪个部分
        if (point[0] > octant->x) mortonCode |= 1;
        if (point[1] > octant->y) mortonCode |= 2;
        if (point[2] > octant->z) mortonCode |= 4;
        if (octant->child[mortonCode])     // 如果子节点存在那么直接读出相关参数
        {
            childOctant = octant->child[mortonCode];
            childExtent = childOctant->extent;
        }
        else
        {
            // 定义子节点宽度
            childExtent = 0.5 * extent;      
            // 更新中心坐标
            static const double factor[] = {-0.5, 0.5};
            double childX = octant->x + factor[(mortonCode & 1) > 0] * extent / 2.0;    // childxyz表示8个子区域的中点
            double childY = octant->y + factor[(mortonCode & 2) > 0] * extent / 2.0;
            double childZ = octant->z + factor[(mortonCode & 4) > 0] * extent / 2.0;
            // 创建新的子节点
            childOctant = new Octant;
            childOctant->x = childX;              // 中心点坐标
            childOctant->y = childY;
            childOctant->z = childZ;
            childOctant->extent = childExtent;    // 节点边长
            childOctant->depth = octant->depth + 1;         // 深度
            octant->child[mortonCode] = childOctant;
            if (childExtent <= params_.minExtent)       // 插入叶子结点
            {
                voxel_->push_back(childOctant);
                childOctant->mortonCode = (morton <<  3) + mortonCode;
            }
        }
        uint32_t size = childOctant->size;
        morton = (morton <<  3) + mortonCode;     // 依次保存点的层次信息morton code
        // 继续调用递归
        insertOctant(point, childOctant, childExtent, morton, voxels);
        if (size != childOctant->size)            // 说明添加了节点
        {
            octant->size += 1;
        }
    }
    else    // 不用分类 直接插入点 这是唯一可以插入points的地方
    {
        if (octant->size == 0)
        {
            for (uint32_t i = 0; i < 26; ++i)   // 遍历周围26个位置
            {
                if (octant->neighbor[i] != nullptr)
                    continue;
                auto neighborCode = neighborVoxels(morton, neighborTable[i], 1, octant->depth);
                // 构造周围节点的voxel
                auto neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);
                // 找到了周围节点
                octant->neighbor[i] = neighborOctant;
                neighborOctant->neighbor[oppNeighborTableID[i]] = octant;
            }
        }

        // 遍历该voxel中的所有点 判断是否满足插入距离
        for (uint32_t i = 0; i < octant->size; ++i)
        {
            const Eigen::Vector3d& p = points[octant->successors_[i]];      // 取出空间点坐标
            // 基于L2距离是比较
            auto dis = L2Distance::compute(point, p);                       // 计算两点之间的距离差异
            if (dis < std::pow(params_.minSize, 2))                         // 比较距离小于阈值
                return;
            // 基于L1距离的比较
            // auto dis = L1Distance::compute(point, p);
            // if (dis < params_.minSize)                         // 比较距离小于阈值
            //     return;

        }
        
        octant->size += 1;
        data_->push_back(point); // 将点云插入到data_中
        octant->successors_.push_back(data_->size()-1);
        // 遍历自身和周围26个voxels并记录下来
        voxels.insert(octant);
        for (uint32_t i = 0; i < 26; ++i)
            voxels.insert(octant->neighbor[i]);
    }
}

/**
 * @description: 创建一个空的voxel 并不包含任何点云数据 主要为了填补表面空缺或者漏洞区域 （带有邻居节点创建）
 * @param {Octant*} octant 当前层级节点指针
 * @param {uint32_t} morton 需要创建节点的morton code
 * @param {int} startDepth 搜索起始深度
 * @param {int} depth 搜索最大深度
 * @return {Octant*} 返回创建后的节点指针
 */
Octree::Octant* Octree::createOctant(Octant* octant, uint32_t morton, uint32_t startDepth, const uint32_t maxDepth)
{
    Octant* childOctant;
    double childExtent;

    if (startDepth > maxDepth)
    {
        octant->mortonCode = morton;
        for (uint32_t i = 0; i < 26; ++i)   // 遍历周围26个位置
        {
            if (octant->neighbor[i] != nullptr)
                continue;
            auto neighborCode = neighborVoxels(morton, neighborTable[i], 1, octant->depth);
            // 构造周围节点的voxel
            auto neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);
            // 找到了周围节点
            octant->neighbor[i] = neighborOctant;
            neighborOctant->neighbor[oppNeighborTableID[i]] = octant;
        }
        return octant;      // 如果到达最深深度就直接返回当前节点指针
    }
    
    if (octant->isLeaf)             
        octant->isLeaf = false;     // 节点分裂 设置为False
    auto mortonCode = (morton >> ((maxDepth-startDepth)*3)) & 0x07;
    if (octant->child[mortonCode] == 0)       // 如果当前节点不存在就创建节点
    {
        // 定义子节点宽度
        childExtent = 0.5 * octant->extent;      
        // 更新中心坐标
        static const double factor[] = {-0.5, 0.5};
        double childX = octant->x + factor[(mortonCode & 1) > 0] * octant->extent / 2.0;    // childxyz表示8个子区域的中点
        double childY = octant->y + factor[(mortonCode & 2) > 0] * octant->extent / 2.0;
        double childZ = octant->z + factor[(mortonCode & 4) > 0] * octant->extent / 2.0;
        // 创建新的子节点
        childOctant = new Octant;
        childOctant->x = childX;              // 中心点坐标
        childOctant->y = childY;
        childOctant->z = childZ;
        childOctant->extent = childExtent;    // 节点边长
        childOctant->depth = octant->depth + 1;         // 深度
        octant->child[mortonCode] = childOctant;
        if (childOctant->depth >= maxDepth)
        {
            voxel_->push_back(childOctant);
        }
    }
    else    // 如果节点存在那么直接调用
    {
        childOctant = octant->child[mortonCode];
        childExtent = childOctant->extent;
    }
    // 继续调用递归
    auto leafOctant = createOctant(childOctant, morton, startDepth+1, maxDepth);

    return leafOctant;
}

/**
 * @description: 创建一个空的voxel 并不包含任何点云数据 主要为了填补表面空缺或者漏洞区域 （不带有邻居节点创建）
 * @param {Octant*} octant 当前层级节点指针
 * @param {uint32_t} morton 需要创建节点的morton code
 * @param {int} startDepth 搜索起始深度
 * @param {int} depth 搜索最大深度
 * @return {Octant*} 返回创建后的节点指针
 */
Octree::Octant* Octree::createOctantSimply(Octant* octant, uint32_t morton, uint32_t startDepth, const uint32_t maxDepth)
{
    Octant* childOctant;
    double childExtent;

    if (startDepth > maxDepth)
    {
        octant->mortonCode = morton;
        return octant;      // 如果到达最深深度就直接返回当前节点指针
    }
    
    auto mortonCode = (morton >> ((maxDepth-startDepth)*3)) & 0x07;
    if (octant->child[mortonCode] == 0)       // 如果当前节点不存在就创建节点
    {
        // 定义子节点宽度
        childExtent = 0.5 * octant->extent;      
        // 更新中心坐标
        static const double factor[] = {-0.5, 0.5};
        double childX = octant->x + factor[(mortonCode & 1) > 0] * octant->extent / 2.0;    // childxyz表示8个子区域的中点
        double childY = octant->y + factor[(mortonCode & 2) > 0] * octant->extent / 2.0;
        double childZ = octant->z + factor[(mortonCode & 4) > 0] * octant->extent / 2.0;
        // 创建新的子节点
        childOctant = new Octant;
        childOctant->x = childX;              // 中心点坐标
        childOctant->y = childY;
        childOctant->z = childZ;
        childOctant->extent = childExtent;    // 节点边长
        childOctant->depth = octant->depth + 1;         // 深度
        octant->child[mortonCode] = childOctant;
        if (childOctant->depth >= maxDepth)
        {
            voxel_->push_back(childOctant);
        }
    }
    else    // 如果节点存在那么直接调用
    {
        childOctant = octant->child[mortonCode];
        childExtent = childOctant->extent;
    }
    // 继续调用递归
    auto leafOctant = createOctantSimply(childOctant, morton, startDepth+1, maxDepth);

    return leafOctant;
}

/**
 * @description: 更新指定点的法向量
 * @param {Vector3d&} camera 相机空间位置
 * @param {std::unordered_set<Octant*>} voxels 需要计算法向量的octants
 * @return {*}
 */
void Octree::updateNormals(const Eigen::Vector3d& camera, std::unordered_set<Octant*> &voxels)
{
    // 定义计算变量
    const std::vector<Eigen::Vector3d> &points = *data_;        // 点云数据
    std::vector<Eigen::Vector3d> &normals = *normal_;           // 点云法向量
    std::vector<bool> &normalConf = *normalConf_;               // 点云法向量置信度
    std::vector<double> &curvature = *curvature_;              // 点云曲率
    
    float sqrRadius = L2Distance::sqr(params_.normalRadius);  // "squared" radius
    std::vector<uint32_t> resultIndices;
    
    // 取出voxels中的所有点
    std::vector<uint32_t> queryIdx;
    for (auto v: voxels) 
    {
        queryIdx.insert(queryIdx.end(), v->successors_.begin(), v->successors_.end());
    }
    
    #pragma omp parallel for private(resultIndices)
    for (uint32_t i = 0; i < queryIdx.size(); ++i) 
    {
        if (normalConf[queryIdx[i]])      // 如果是有效的法向量，那么就不再重新计算了
            continue;
        resultIndices.clear();
        // 查询节点邻居
        radiusNeighbors(root_, points[queryIdx[i]], params_.normalRadius, sqrRadius, resultIndices);
        // 判断改点是否为有效点 如果其周围点数量超过阈值那么就是有效点
        if (resultIndices.size() < params_.pointsValid)
            continue;
        
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> A = Eigen::Matrix<double, 3, 3, Eigen::DontAlign>::Zero();
        Eigen::Matrix<double, 3, 1, Eigen::DontAlign> b = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>::Zero();
        for (uint32_t j = 0; j < resultIndices.size(); ++j) 
        {
            Eigen::Vector3d vtmp = points[resultIndices[j]];
            A += vtmp * vtmp.transpose();
            b += vtmp;
        }
        A += points[queryIdx[i]]*points[queryIdx[i]].transpose();
        b += points[queryIdx[i]];
        int cnt = (int)(resultIndices.size()) + 1;
        b = b / cnt;
        A = A / cnt - b * b.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3, Eigen::DontAlign>> eigenSolver;
        eigenSolver.computeDirect(A, Eigen::ComputeEigenvectors);
        // PCA find normal
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> eivecs;
        Eigen::Matrix<double, 3, 1, Eigen::DontAlign> eivals;
        eivecs = eigenSolver.eigenvectors();
        eivals = eigenSolver.eigenvalues();
        
        double curve = eivals(0) / (eivals(0) + eivals(1) + eivals(2) + 1e-9);      
        Eigen::Matrix<double, 3, 1, Eigen::DontAlign> normal = eivecs.col(0);

        // 更新法向量正负 相机所在方向法向量为正
        Eigen::Vector3d orientation = camera - points[queryIdx[i]];
        if (normal.norm() == 0.0) 
        {
            normal = orientation;
            if (normal.norm() == 0.0) 
            {
                normal = Eigen::Vector3d(0.0, 0.0, 1.0);
            } else 
            {
                normal.normalize();
            }
            
        } else if (normal.dot(orientation) < 0.0) 
        {
            normal *= -1.0;
        }
        
        // 保存最终结果
        normals[queryIdx[i]] = normal;
        normalConf[queryIdx[i]] = true;
        curvature[queryIdx[i]] = curve;
    }
}

/**
 * @description: 更新当前节点的平均曲率
 * @param {Octant*} octant 当前节点
 * @return {*}
 */
void Octree::updateCurvature(Octant* octant)
{
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    // 1. 遍历voxel所有点的法向量构造协方差矩阵
    // 定义计算变量
    double centroid0 = 0;
    double centroid1 = 0;
    double centroid2 = 0;
    double C_00 = 0;
    double C_01 = 0;
    double C_02 = 0;
    double C_10 = 0;
    double C_11 = 0;
    double C_12 = 0;
    double C_20 = 0;
    double C_21 = 0;
    double C_22 = 0;

    for (uint32_t i = 0; i < octant->size; ++i)
    {
        uint32_t idx = octant->successors_[i];
        centroid0 += normals[idx][0];
        centroid1 += normals[idx][1];
        centroid2 += normals[idx][2];
    }
    centroid0 /= octant->size;
    centroid1 /= octant->size;
    centroid2 /= octant->size;

    for (uint32_t i = 0; i < octant->size; ++i)
    {
        uint32_t idx = octant->successors_[i];
        C_00 += (normals[idx][0] - centroid0) * (normals[idx][0] - centroid0);
        C_01 += (normals[idx][0] - centroid0) * (normals[idx][1] - centroid1);
        C_02 += (normals[idx][0] - centroid0) * (normals[idx][2] - centroid2);
        C_11 += (normals[idx][1] - centroid1) * (normals[idx][1] - centroid1);
        C_12 += (normals[idx][1] - centroid1) * (normals[idx][2] - centroid2);
        C_22 += (normals[idx][2] - centroid2) * (normals[idx][2] - centroid2);
    }
    C_00 /= octant->size;
    C_01 /= octant->size;
    C_02 /= octant->size;
    C_10 = C_01;
    C_11 /= octant->size;
    C_12 /= octant->size;
    C_20 = C_02;
    C_21 = C_12;
    C_22 /= octant->size;

    Eigen::Matrix3d C;
        C << C_00, C_01, C_02, C_10, C_11, C_12, C_20, C_21, C_22;
    Eigen::Vector3d eigenvalue;
    computeEigen33Values(C, eigenvalue);
    if (eigenvalue[2] == 0)
        octant->curvature = 0;
    else
        octant->curvature = eigenvalue[0]/eigenvalue[2];        // 用最小特征值/最大特征值作为曲率
}
/************************************ END ************************************/

/**************************** Octree 搜索相关函数 ******************************/

/**
 * @description: 搜索子节点中的叶节点中的successors_指针
 * @param {Octant*} octant 从当前节点开始搜索子节点
 * @param {vector<uint32_t>} successors 用于保存结果的vector
 * @return {*}
 */
void Octree::subSuccessors(const Octant* octant, std::vector<uint32_t> &successors)
{
    if (octant == nullptr)
        return;
    if (!(octant->isLeaf))
    {
        for (uint32_t c = 0; c < 8; ++c)
        {
            subSuccessors(octant->child[c], successors);
        }
    }
    else
    {
        successors.insert(successors.end(), octant->successors_.begin(), octant->successors_.end());
    }
}

/**
 * @description: 判断查询点query及其搜索半径是否完全被节点o包含
 * @param {Vector3d&} query 查询点
 * @param {double} sqRadius 搜索平方半径
 * @param {Octant*} o 当前节点
 * @return {*}
 */
bool Octree::contains(const Eigen::Vector3d& query, double sqRadius, const Octant* o)
{
    // we exploit the symmetry to reduce the test to test
    // whether the farthest corner is inside the search ball.
    double x = query[0] - o->x;
    double y = query[1] - o->y;
    double z = query[2] - o->z;

    x = std::abs(x);
    y = std::abs(y);
    z = std::abs(z);
    // reminder: (x, y, z) - (-e, -e, -e) = (x, y, z) + (e, e, e)
    x += (o->extent)/2.0f;
    y += (o->extent)/2.0f;
    z += (o->extent)/2.0f;

    return (L2Distance::norm(x, y, z) < sqRadius);
}

/**
 * @description: 判断查询点query及其搜索半径是否与节点o范围有重合
 * @param {Vector3d&} query 查询点
 * @param {double} radius 搜索半径
 * @param {double} sqRadius 搜索平方半径
 * @param {Octant*} o 当前节点
 * @return {*}
 */
bool Octree::overlaps(const Eigen::Vector3d& query, double radius, double sqRadius, const Octant* o)
{
    // we exploit the symmetry to reduce the test to testing if its inside the Minkowski sum around the positive quadrant.
    double x = query[0] - o->x;
    double y = query[1] - o->y;
    double z = query[2] - o->z;

    x = std::abs(x);
    y = std::abs(y);
    z = std::abs(z);

    double maxdist = radius + (o->extent)/2.0f;

    // Completely outside, since q' is outside the relevant area.   如果voxel完全在搜索半径外面 那么返回false
    if (x > maxdist || y > maxdist || z > maxdist) return false;

    int32_t num_less_extent = (x < (o->extent)/2.0f) + (y < (o->extent)/2.0f) + (z < (o->extent)/2.0f);    // 判断三个坐标轴坐标有几个坐标在范围内

    // Checking different cases:

    // a. inside the surface region of the octant. 搜索点query落在这个voxel内
    if (num_less_extent > 1) return true;

    // b. checking the corner region && edge region.
    x = std::max(x - (o->extent)/2.0f, 0.0);
    y = std::max(y - (o->extent)/2.0f, 0.0);
    z = std::max(z - (o->extent)/2.0f, 0.0);

    return (L2Distance::norm(x, y, z) < sqRadius);
}

/**
 * @description: 判断查询点query及其搜索半径是否完全被节点o包含
 * @param {Vector3d&} query 查询点
 * @param {double} radius 搜索半径
 * @param {Octant*} octant 当前节点
 * @return {*}
 */
bool Octree::inside(const Eigen::Vector3d& query, double radius, const Octant* octant)
{
    // we exploit the symmetry to reduce the test to test
    // whether the farthest corner is inside the search ball.
    double x = query[0] - octant->x;
    double y = query[1] - octant->y;
    double z = query[2] - octant->z;

    x = std::abs(x) + radius;
    y = std::abs(y) + radius;
    z = std::abs(z) + radius;

    if (x > (octant->extent)/2.0f) 
        return false;
    if (y > (octant->extent)/2.0f) 
        return false;
    if (z > (octant->extent)/2.0f) 
        return false;

    return true;
}

/**
 * @description: 自上而下在查询点的指定半径内搜索其邻居点
 * @param {Octant*} octant 当前搜索Octree节点
 * @param {Vector3d&} query 查询点坐标
 * @param {double} radius    搜索半径
 * @param {double} sqrRadius 搜索平方半径
 * @param {vector<uint32_t>&} resultIndices 保存查询结果
 * @return {*}
 */
void Octree::radiusNeighbors(const Octant* octant, const Eigen::Vector3d& query, double radius,
                                                 double sqrRadius, std::vector<uint32_t>& resultIndices) const
{
    const std::vector<Eigen::Vector3d>& points = *data_;

    // 第一种情况 如果voxel的最远点在radius半径内 也就是整个voxel都在搜索半径内 那么我们直接将voxel的所有点push到resultIndices和distances中
    if (contains(query, sqrRadius, octant))             // 给定 查找的点云坐标query 查找半径平方sqrRadius 以及voxel的中心点octant 查找这个voxel的最远点是否在搜索半径内
    {
        std::vector<uint32_t> pointVec;
        subSuccessors(octant, pointVec);
        for (uint32_t i = 0; i < pointVec.size(); ++i)     // octant->size 表示这个voxel中包含多少点云
        {
            resultIndices.push_back(pointVec[i]);          // 保存节点index
        }
        return;  // 嵌套退出
    }

    // 第二种情况 如果voxel是叶节点 但voxel并不在搜索半径内 逐个搜索节点仅保存在搜索半径的点
    if (octant->isLeaf)     // 如果这个voxel的最远点不在搜索半径内 但是叶子节点
    {
        for (uint32_t i = 0; i < octant->size; ++i)
        {
            uint32_t pointIdx = octant->successors_[i];     // 读取节点中点云的idx
            const Eigen::Vector3d& p = points[pointIdx];    // 子节点的坐标
            double dist = L2Distance::compute(query, p);     // 计算子节点的坐标和查询点坐标之间的距离
            if (dist < sqrRadius)       // 仅保存距离小于搜索半径的点
            {
                resultIndices.push_back(pointIdx);
            }
        }
        return;     // 嵌套退出
    }

    // 第三种情况 如果该voxel即不完全在搜索半径内 而且也不是叶子voxel 因此要进一步将voxel分为8块依次搜索每一块内的子voxel
    for (uint32_t c = 0; c < 8; ++c)
    {
        if (octant->child[c] == 0) continue;
        if (!overlaps(query, radius, sqrRadius, octant->child[c])) continue;
        radiusNeighbors(octant->child[c], query, radius, sqrRadius, resultIndices);
    }
}

/**
 * @description: 自上而下搜索距离查询点最近的邻居节点
 * @param {Octant*} octant 当前节点
 * @param {Vector3d&} query 查询点坐标
 * @param {double} minDistance 查询范围最小距离
 * @param {double&} maxDistance 查询范围最大距离
 * @param {uint32_t&} resultIndex 保存查询结果
 * @return {*}
 */
bool Octree::findNeighbor(const Octant* octant, const Eigen::Vector3d& query, double minDistance,
                                              double& maxDistance, uint32_t& resultIndex) const
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const std::vector<bool>& normalConf = *normalConf_;

    // 1. first descend to leaf and check in leafs points.
    if (octant->isLeaf)
    {
        double sqrMaxDistance = L2Distance::sqr(maxDistance);
        double sqrMinDistance = (minDistance < 0) ? minDistance : L2Distance::sqr(minDistance);
        uint32_t resultIndexInit = resultIndex;

        for (uint32_t i = 0; i < octant->size; ++i)
        {
            uint32_t pointIdx = octant->successors_[i];     // 读取节点中点云的idx
            // 判断有效点 如果搜索到的最近点不是有效点那么要继续搜索
            if (normalConf[pointIdx])
            {
                const Eigen::Vector3d& p = points[pointIdx];    // 子节点的坐标
                double dist = L2Distance::compute(query, p);
                if (dist > sqrMinDistance && dist < sqrMaxDistance)
                {
                    resultIndex = pointIdx;
                    sqrMaxDistance = dist;
                }
            }
            
        }

        if (resultIndexInit == resultIndex)
            return false;
        
        maxDistance = L2Distance::sqrt(sqrMaxDistance);
        return inside(query, maxDistance, octant);
    }

    // determine Morton code for each point...
    uint32_t mortonCode = 0;
    if (query[0] > octant->x) mortonCode |= 1;
    if (query[1] > octant->y) mortonCode |= 2;
    if (query[2] > octant->z) mortonCode |= 4;

    if (octant->child[mortonCode] != 0)
    {
        if (findNeighbor(octant->child[mortonCode], query, minDistance, maxDistance, resultIndex)) 
            return true;
    }

    // 2. if current best point completely inside, just return.
    double sqrMaxDistance = L2Distance::sqr(maxDistance);

    // 3. check adjacent octants for overlap and check these if necessary.
    for (uint32_t c = 0; c < 8; ++c)
    {
        if (c == mortonCode) continue;
        if (octant->child[c] == 0) continue;
        if (!overlaps(query, maxDistance, sqrMaxDistance, octant->child[c])) continue;
        if (findNeighbor(octant->child[c], query, minDistance, maxDistance, resultIndex))
            return true;  // early pruning
    }

    // all children have been checked...check if point is inside the current octant...
    return inside(query, maxDistance, octant);
}
/************************************ END ************************************/

/**************************** 模型网格生成以及分裂 *****************************/
/**
 * @description: 删除octant中的顶点 对应verts中指定的index顶点
 * @param {Octant*} octant 当前节点
 * @return {*}
 */
void Octree::trianglesClear(Octant* octant)
{
    VertsArray& verts = *verts_;
    for (std::vector<Eigen::Vector3i>::iterator it = octant->triangles->begin(); it != octant->triangles->end(); ++it)
    {
        Eigen::Vector3i& triangles = *it;
        for (uint32_t k = 0; k < 3 ; ++k)   // 遍历这个三角形的三个顶点idx
        {
            verts.remove(triangles[k], octant->mortonCode);
        }
    }
    octant->triangles->clear();
}

/**
 * @description: sdf差值函数 marchingCubes中使用
 * @return {*}
 */
static inline Eigen::Vector3d sdfInterp(const Eigen::Vector3d p1, const Eigen::Vector3d p2, double valp1, double valp2) 
{
    // if (fabs(0.0f - valp1) < 1.0e-5f) return p1;
	// if (fabs(0.0f - valp2) < 1.0e-5f) return p2;
	// if (fabs(valp1 - valp2) < 1.0e-5f) return p1;

	double w2 = (0.0 - valp1) / (valp2 - valp1);
	double w1 = 1 - w2;

	return Eigen::Vector3d(p1[0] * w1 + p2[0] * w2,
                          p1[1] * w1 + p2[1] * w2,
                          p1[2] * w1 + p2[2] * w2);
}

/**
 * @description: 对八叉树中的每个叶子结点的voxel执行MarchingCubes
 * @param {Octant*} octant 八叉树叶子结点voxel指针
 * @param {Eigen::Vector3i*} valid_cords 表示叶节点中voxel的尺寸 如果是1就表示没有分裂
 * @param {Eigen::Tensor<double, 3>} dense_sdf 大小可变的保存SDF数组 大小与valid_cords大小相关
 * @param {vector<bool>} mask 标记不需要构建三角面的subCubes序号
 * @param {uint32_t} level 分裂层级
 * @param {Eigen::Vector3d} offsets 
 * @return {*}
 */
void Octree::marchingCubesSparse(Octant* octant, const Eigen::Vector3i* valid_cords, const Eigen::Tensor<double, 3> &dense_sdf,
                                 const std::vector<bool> mask, const uint32_t level,const Eigen::Vector3d offsets)
{
    // 判断该octant是否已经初始化了triangles Vector容器
    if (octant->triangles == nullptr)
        octant->triangles = new std::vector<Eigen::Vector3i>;       // 为每一个叶子voxel分配三角形顶点容器
    else
    {
        if (!params_.meshOverlap)               // 如果不允许重复面那么在重新构造面的时候就要删除
            trianglesClear(octant);
    }
        
    VertsArray& verts = *verts_;
    // 遍历该节点在周围邻居 将verts的face保存下来
    std::vector<Eigen::Vector3i> neighborTriangles;
    neighborTriangles.insert(neighborTriangles.end(), octant->triangles->begin(), octant->triangles->end());
    for (uint32_t i = 0; i < 26; ++i)
    {
        auto neighborOctant = octant->neighbor[i];
        if (neighborOctant == 0 || neighborOctant->triangles == 0 || neighborOctant->triangles->size() == 0)
            continue;
        neighborTriangles.insert(neighborTriangles.end(), neighborOctant->triangles->begin(), neighborOctant->triangles->end());
    }
    // 使用MarchingDense函数 生成多个mesh顶点
    std::vector<Eigen::Vector3d> vp;        // 用于保存MarchingDense函数生成的顶点
    marchingCubesDense(vp, valid_cords, dense_sdf, mask, level*level*level, offsets, octant->extent/level);
    // 筛选输出的顶点去除重复和错误点
    for (uint32_t i = 0; i < vp.size()/3; ++i)
    {
        Eigen::Vector3i vt;         // 用于保存faceID
        for (uint32_t j = 0; j < 3; ++j)
        {
            // 遍历该节点的周围其他节点的verts 判断是否是重复的顶点
            bool isRepeat = false;
            uint32_t repeatNum;
            for (uint32_t it = 0; it < neighborTriangles.size(); ++it)
            {
                Eigen::Vector3i triangles = neighborTriangles[it];   // 取出一个triangles（包含三角形三个顶点idx）
                for (uint32_t k = 0; k < 3 ; ++k)   // 遍历这个三角形的三个顶点idx
                {
                    // 由于sdfInterp中浮点数运输会引入舍入误差所以不采用 // if (vp[vi] == verts[triangles[k]].point)
                    // 使用近似相等判断
                    if ((std::abs(vp[3*i+j][0] - verts[triangles[k]].point[0]) < 1e-4) && (std::abs(vp[3*i+j][1] - verts[triangles[k]].point[1]) < 1e-4) && (std::abs(vp[3*i+j][2] - verts[triangles[k]].point[2]) < 1e-4))
                    {
                        repeatNum = triangles[k];
                        isRepeat = true;
                        break;
                    }
                }
                if (isRepeat)
                    break;
            }
            if (isRepeat)
            {
                if (verts.id_update(repeatNum, vp[3*i+j], octant->mortonCode))
                {
                    vt[j] = repeatNum;
                }
            }
            else
            {
                vt[j] = verts.raw_insert(vp[3*i+j], octant->mortonCode);
            }
        }
        // 去除退化的face 比如 17 17 5 或者 248 65 248 要删除这种面
        if (((std::abs(vp[3*i][0] - vp[3*i+1][0]) < 1e-5) && (std::abs(vp[3*i][1] - vp[3*i+1][1]) < 1e-5) && (std::abs(vp[3*i][2] - vp[3*i+1][2]) < 1e-5)) ||
            ((std::abs(vp[3*i][0] - vp[3*i+2][0]) < 1e-5) && (std::abs(vp[3*i][1] - vp[3*i+2][1]) < 1e-5) && (std::abs(vp[3*i][2] - vp[3*i+2][2]) < 1e-5)) ||
            ((std::abs(vp[3*i+1][0] - vp[3*i+2][0]) < 1e-5) && (std::abs(vp[3*i+1][1] - vp[3*i+2][1]) < 1e-5) && (std::abs(vp[3*i+1][2] - vp[3*i+2][2]) < 1e-5)))
        {
            verts.remove(vt[0], octant->mortonCode);
            verts.remove(vt[1], octant->mortonCode);
            verts.remove(vt[2], octant->mortonCode);
            continue;
        }
        octant->triangles->push_back(vt);
        neighborTriangles.push_back(vt);        // 重要！
    }
}

/**
 * @description: 从点云中生成模型三角面主函数 包括计算Cubes顶点的SDF值
 * @param {std::unordered_set<Octant*>} voxels 需要Marching Cubes的octants
 * @return {*}
 */
void Octree::updateVoxel(std::unordered_set<Octant*> &voxels)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    const std::vector<bool> &normalConf = *normalConf_;
    const std::vector<double> &curvature = *curvature_;

    std::vector<Octant*> voxelsList;
    voxelsList.insert(voxelsList.end(), voxels.begin(), voxels.end());     // c++17 特性将unordered_set迁移到voxelsList

    // 第一次处理所有叶节点包括：计算叶节点cubes的顶点SDF 然后生成三角面 这一步同时要对曲率和边界位置进行判定
    #pragma omp parallel for
    for (uint32_t i = 0; i < voxelsList.size(); ++i)
    {
        Octant* childOctant = voxelsList[i];
        double cubeSDF[8];                             // 保存sdf值
        double cubeValid[8];                           // 指示该顶点是否有效

        // 计算叶节点边长和深度
        double extent = extent_;         // octree最大边长
        while(extent > params_.minExtent)
        {
            extent /= 2.0;
        }

        // 计算平均曲率
        if (childOctant->size > 0)
        {
            double sumCurvature = 0;
            uint32_t count = 0;
            for (const auto pointIdx : childOctant->successors_)
            {
                if (normalConf[pointIdx])       // 如果该点有效
                {
                    sumCurvature += curvature[pointIdx];
                    count++;
                }
            }
            childOctant->curvature = sumCurvature / (double)count;
        }

        // 遍历voxel的8个顶点 以左上角为起始点 查询最近点计算SDF值
        for (int a = 0; a < 2; ++a)         // x方向
        {
            double x = childOctant->x + (a % 2 == 0 ? -1 : 1) * extent / 2.0;
            for (int b = 0; b < 2; ++b)     // y方向
            {
                double y = childOctant->y + (b % 2 == 0 ? -1 : 1) * extent / 2.0;
                for (int c = 0; c < 2; ++c) // z方向
                {
                    double z = childOctant->z + (c % 2 == 0 ? -1 : 1) * extent / 2.0;

                    // 计算voxel顶点的最近邻
                    double distances = INFINITY;
                    uint32_t nearIdx = 0;
                    findNeighbor(root_, Eigen::Vector3d(x, y, z), -1, distances, nearIdx);
                    // 利用法向量计算点到点云平面的距离
                    // distances 为查询点到最近邻点的L2直线距离
                    // distance 为查询点到最近邻点的法向量距离
                    auto nearPoint = points[nearIdx];   // 最近距离的点云
                    auto nearNormal = normals[nearIdx]; // 最近距离的点云的法向量
                    double distance = (Eigen::Vector3d(x, y, z) - nearPoint).dot(nearNormal);
                    if (std::abs(distances) < params_.sdfRadius)
                    {
                        cubeSDF[4*a+2*b+c] = distance;
                        cubeValid[4*a+2*b+c] = 1;
                    }
                    else
                    {
                        cubeValid[4*a+2*b+c] = 0;              // 如果查询点距离最近点大于阈值sdf_radius 那么设置为无效
                    }
                }
            }
        }
        std::copy(cubeSDF, cubeSDF + 8, childOctant->sdf);
        std::copy(cubeValid, cubeValid + 8, childOctant->weight);       // 临时帧的octree weight用于表示该点是否有效
    }
}

/**
 * @description: 从每个临时的树中遍历所有的voxel 将点云曲率等信息插入到本树中
 * @param {Octree} &tree 临时帧对应的树
 * @param {Octant} *baseOctant 本树当前对应的Octant指针
 * @param {Octant} *treeOctant 临时树当前对应的Octant指针
 * @return {*}
 */
void Octree::fusionPoints(Octree &tree, Octant *baseOctant, Octant *treeOctant)
{
    const std::vector<Eigen::Vector3d> &treePoints = *(tree.data_);        // 点云数据
    const std::vector<Eigen::Vector3d> &treeNormals = *(tree.normal_);           // 点云法向量
    const std::vector<bool> &treeNormalConf = *(tree.normalConf_);           // 点云法向量置信度
    const std::vector<Eigen::Vector3d> &basePoints = *data_;        // 点云数据
    // 插入点数据
    for (uint32_t i = 0; i < treeOctant->size; ++i) 
    {
        auto point = treePoints[treeOctant->successors_[i]];
        auto normal = treeNormals[treeOctant->successors_[i]];
        auto normalConf = treeNormalConf[treeOctant->successors_[i]];
        // 遍历该voxel中的所有点 判断是否满足插入距离
        bool isValid = true;
        for (uint32_t j = 0; j < baseOctant->size; ++j)
        {
            const Eigen::Vector3d& p = basePoints[baseOctant->successors_[j]];      // 取出空间点坐标
            // 基于L2距离是比较
            auto dis = L2Distance::compute(point, p);                       // 计算两点之间的距离差异
            if (dis < std::pow(params_.minSize, 2))                         // 比较距离小于阈值
            {
                isValid = false;
                break;
            }
            // 基于L1距离的比较
            // auto dis = L1Distance::compute(point, p);
            // if (dis < params_.minSize)                         // 比较距离小于阈值
            // {
            //     isValid = false;
            //     break;
            // }

        }
        if (isValid)
        {
            baseOctant->size += 1;
            data_->push_back(point);
            normal_->push_back(normal);
            normalConf_->push_back(normalConf);
            baseOctant->successors_.push_back(data_->size()-1);
        }
    }
    // 融合插入曲率信息
    if (baseOctant->curveWeight == 0)       // 第一次插入曲率
    {
        baseOctant->curvature = treeOctant->curvature;
        baseOctant->curveWeight = treeOctant->size;
    }
    else                                    // 融合更新
    {
        baseOctant->curvature = ((baseOctant->curveWeight * baseOctant->curvature) + (treeOctant->size * treeOctant->curvature)) \
                                / (baseOctant->curveWeight + treeOctant->size);
        baseOctant->curveWeight += treeOctant->size;
    }
}


/**
 * @description: 用于判断当前voxel是否需要裁剪 如果需要裁剪将裁剪mask保存在subMask中
 * @param {Octant*} octant 当前叶子结点voxel
 * @param {vector<bool>} &subMask 标识哪些subCube需要裁剪
 * @param {int} level 分裂等级
 * @return {bool} 返回是否需要裁剪该voxel
 */
bool Octree::maskEdgeVoxel(Octant* octant, std::vector<bool> &subMask, int level)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const VertsArray& verts = *verts_;
    std::unordered_set<uint32_t> faceIds;
    bool isDetial = true;
    // 收集当前voxel中的所有mesh顶点序号
    for (uint32_t i = 0; i < octant->triangles->size(); ++i)
    {
        faceIds.insert((*(octant->triangles))[i][0]);
        faceIds.insert((*(octant->triangles))[i][1]);
        faceIds.insert((*(octant->triangles))[i][2]);
    }
    // 遍历所有mesh顶点与当前voxel和周围voxel邻居的点云进行距离比较判断是否距离过远
    for (auto fid : faceIds)
    {
        isDetial = true;
        auto vt = verts[fid].point;
        for (uint32_t i = 0; i < octant->size; ++i)        // 遍历voxel中的每一个点
        {
            auto idx =  octant->successors_[i];
            float dis = L2Distance::compute(vt, points[idx]);
            if (dis < std::pow(params_.minBorder, 2))
            {
                isDetial = false;
                break;
            }
        }
        if (isDetial)
        {
            for (uint32_t i = 0; i < 26; ++i)
            {
                auto neighborOctant = octant->neighbor[i];
                for (uint32_t j = 0; j < neighborOctant->size; ++j)        // 遍历邻居voxel中的每一个点
                {
                    auto idx =  neighborOctant->successors_[j];
                    float dis = L2Distance::compute(vt, points[idx]);
                    if (dis < std::pow(params_.minBorder, 2))
                    {
                        isDetial = false;
                        break;
                    }
                }
                if (!isDetial)
                    break;
            }
        }
        if (isDetial)
            break;
    }
    if (isDetial)       // 表示是否需要裁剪
    {
        // 查找并删除无效的voxel
        Eigen::Vector3d offsets(octant->x-(octant->extent/2), octant->y-(octant->extent/2), octant->z-(octant->extent/2));
        double scale = octant->extent / level;
        for (uint32_t i = 0; i < octant->size; ++i)
        {
            auto p = points[octant->successors_[i]];
            auto op = (p - offsets) / scale;
            int cellX = static_cast<int>(op[0]);
            int cellY = static_cast<int>(op[1]);
            int cellZ = static_cast<int>(op[2]);
            if (cellX >= 0 && cellX < level && cellY >= 0 && cellY < level && cellZ >= 0 && cellZ < level)      // 点坐标有效
            {
                subMask[cellX * level * level + cellY * level + cellZ] = false;
            }
        }
        return true;
    }
    return false;
}

/**
 * @description: 对voxel内按照level层级进行划分为level*level*level个小cube
 * @param {Octant*} octant 当前叶节点voxel
 * @param {Vector3i} *subIndex 保存用于mc的相对顶点坐标
 * @param {Tensor<double, 3>} &subSDF 保存用于mc的SDF值
 * @param {int} level 分裂层级
 * @return {*}
 */
void Octree::SDFdeviation(const Octant* octant, Eigen::Vector3i *subIndex, Eigen::Tensor<double, 3> &subSDF, int level)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    // 八个顶点值
    subSDF(0, 0, 0) = octant->sdf[0];
    subSDF(0, 0, level) = octant->sdf[1];
    subSDF(0, level, 0) = octant->sdf[2];
    subSDF(0, level, level) = octant->sdf[3];
    subSDF(level, 0, 0) = octant->sdf[4];
    subSDF(level, 0, level) = octant->sdf[5];
    subSDF(level, level, 0) = octant->sdf[6];
    subSDF(level, level, level) = octant->sdf[7];
    if (params_.curvatureTHR == -1 ||octant->curvature < params_.curvatureTHR)       // 曲率没有超过阈值 直接三线性插值
    {
        // 遍历sub Cubes的顶点计算其SDF值
        for (int a = 0; a <= level; ++a) {
            for (int b = 0; b <= level; ++b) {
                for (int c = 0; c <= level; ++c) {
                    if (a != level && b != level && c != level)
                        subIndex[a * level * level + b * level + c] = Eigen::Vector3i(a, b, c); // 保存子cube的index序号

                    // 根据相邻顶点的值进行线性插值计算
                    double x_ratio = (double)a / level;
                    double y_ratio = (double)b / level;
                    double z_ratio = (double)c / level;

                    subSDF(a, b, c) = (1 - x_ratio) * (1 - y_ratio) * (1 - z_ratio) * subSDF(0, 0, 0)
                                    + x_ratio * (1 - y_ratio) * (1 - z_ratio) * subSDF(level, 0, 0)
                                    + (1 - x_ratio) * y_ratio * (1 - z_ratio) * subSDF(0, level, 0)
                                    + x_ratio * y_ratio * (1 - z_ratio) * subSDF(level, level, 0)
                                    + (1 - x_ratio) * (1 - y_ratio) * z_ratio * subSDF(0, 0, level)
                                    + x_ratio * (1 - y_ratio) * z_ratio * subSDF(level, 0, level)
                                    + (1 - x_ratio) * y_ratio * z_ratio * subSDF(0, level, level)
                                    + x_ratio * y_ratio * z_ratio * subSDF(level, level, level);
                }
            }
        }
    }
    else        // 曲率超过阈值 其中的部分点需要重新计算细分处的SDF
    {
        // 遍历sub Cubes的顶点计算其SDF值
        for (int a = 0; a <= level; ++a) {
            for (int b = 0; b <= level; ++b) {
                for (int c = 0; c <= level; ++c) {
                    if (a != level && b != level && c != level)
                        subIndex[a * level * level + b * level + c] = Eigen::Vector3i(a, b, c); // 保存子cube的index序号
                    // 判断voxel中的哪些点需要重新计算SDF 一共分为 6面+12边+8顶点+内部区域
                    bool reCal = true;          // 是否需要重新计算SDF
                    if (a == level && b != 0 && b != level && c != 0 && c != level)     // face 1
                    {
                        if (octant->neighbor[0]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == 0 && b != 0 && b != level && c != 0 && c != level)     // face 2
                    {
                        if (octant->neighbor[4]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b == 0 && c != 0 && c != level)     // face 3
                    {
                        if (octant->neighbor[6]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b == level && c != 0 && c != level)     // face 4
                    {
                        if (octant->neighbor[2]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b != 0 && b != level && c == 0)     // face 5
                    {
                        if (octant->neighbor[17]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b != 0 && b != level && c == level)     // face 6
                    {
                        if (octant->neighbor[12]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == level && b == level && c != 0 && c != level)     // edge 1
                    {
                        if (octant->neighbor[0]->curvature < 0.01 || octant->neighbor[1]->curvature < 0.01 || octant->neighbor[2]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == 0 && b == level && c != 0 && c != level)     // edge 2
                    {
                        if (octant->neighbor[2]->curvature < 0.01 || octant->neighbor[3]->curvature < 0.01 || octant->neighbor[4]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == 0 && b == 0 && c != 0 && c != level)     // edge 3
                    {
                        if (octant->neighbor[4]->curvature < 0.01 || octant->neighbor[5]->curvature < 0.01 || octant->neighbor[6]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == level && b == 0 && c != 0 && c != level)     // edge 4
                    {
                        if (octant->neighbor[6]->curvature < 0.01 || octant->neighbor[7]->curvature < 0.01 || octant->neighbor[0]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b == 0 && c == 0)     // edge 5
                    {
                        if (octant->neighbor[6]->curvature < 0.01 || octant->neighbor[16]->curvature < 0.01 || octant->neighbor[17]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b == level && c == 0)     // edge 6
                    {
                        if (octant->neighbor[17]->curvature < 0.01 || octant->neighbor[14]->curvature < 0.01 || octant->neighbor[2]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b == level && c == level)     // edge 7
                    {
                        if (octant->neighbor[2]->curvature < 0.01 || octant->neighbor[9]->curvature < 0.01 || octant->neighbor[12]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a != 0 && a != level && b == 0 && c == level)     // edge 8
                    {
                        if (octant->neighbor[12]->curvature < 0.01 || octant->neighbor[11]->curvature < 0.01 || octant->neighbor[6]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == 0  && b != 0 && b != level && c == 0)     // edge 9
                    {
                        if (octant->neighbor[17]->curvature < 0.01 || octant->neighbor[15]->curvature < 0.01 || octant->neighbor[4]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == level  && b != 0 && b != level && c == 0)     // edge 10
                    {
                        if (octant->neighbor[17]->curvature < 0.01 || octant->neighbor[13]->curvature < 0.01 || octant->neighbor[0]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == level  && b != 0 && b != level && c == level)     // edge 11
                    {
                        if (octant->neighbor[0]->curvature < 0.01 || octant->neighbor[8]->curvature < 0.01 || octant->neighbor[12]->curvature < 0.01)
                            reCal = false;
                    }
                    else if (a == 0  && b != 0 && b != level && c == level)     // edge 12
                    {
                        if (octant->neighbor[12]->curvature < 0.01 || octant->neighbor[10]->curvature < 0.01 || octant->neighbor[4]->curvature < 0.01)
                            reCal = false;
                    }
                    else if ((a == 0  && b == 0 && c == 0) || (a == 0  && b == 0 && c == level) || (a == 0  && b == level && c == 0) || (a == 0  && b == level && c == level) \
                        || (a == level  && b == 0 && c == 0) || (a == level  && b == 0 && c == level) || (a == level  && b == level && c == 0) || (a == level  && b == level && c == level))        // 8个顶点已经赋值不需要再计算
                        continue;
                    // reCal为True表示要重计算SDF
                    if (reCal)
                    {
                        auto orign = Eigen::Vector3d(octant->x - (octant->extent / 2), octant->y - (octant->extent / 2), octant->z - (octant->extent / 2));
                        double subExtent = octant->extent / level;
                        auto query = Eigen::Vector3d(orign[0] + a*subExtent, orign[1] + b*subExtent, orign[2] + c*subExtent);
                        // 查找最近邻
                        double distances = INFINITY;
                        uint32_t nearIdx = 0;
                        findNeighbor(root_, query, -1, distances, nearIdx);
                        auto nearPoint = points[nearIdx];   // 最近距离的点云
                        auto nearNormal = normals[nearIdx]; // 最近距离的点云的法向量
                        subSDF(a, b, c) = (query - nearPoint).dot(nearNormal);
                    }
                    else    // 否则直接三线性插值
                    {
                        // 根据相邻顶点的值进行线性插值计算
                        double x_ratio = (double)a / level;
                        double y_ratio = (double)b / level;
                        double z_ratio = (double)c / level;

                        subSDF(a, b, c) = (1 - x_ratio) * (1 - y_ratio) * (1 - z_ratio) * subSDF(0, 0, 0)
                                        + x_ratio * (1 - y_ratio) * (1 - z_ratio) * subSDF(level, 0, 0)
                                        + (1 - x_ratio) * y_ratio * (1 - z_ratio) * subSDF(0, level, 0)
                                        + x_ratio * y_ratio * (1 - z_ratio) * subSDF(level, level, 0)
                                        + (1 - x_ratio) * (1 - y_ratio) * z_ratio * subSDF(0, 0, level)
                                        + x_ratio * (1 - y_ratio) * z_ratio * subSDF(level, 0, level)
                                        + (1 - x_ratio) * y_ratio * z_ratio * subSDF(0, level, level)
                                        + x_ratio * y_ratio * z_ratio * subSDF(level, level, level);
                    }
                }
            }
        }
    }
}

/**
 * @description: 在fusionVoxel函数前用于修改Octree中的voxel的Fixed标识位；用于填补多帧之间的mesh间隙；用在直出非加权的mode中
 * @param {Octree} &tree    每一帧的Octree
 * @return {*}
 */
void Octree::voxelStatusCheck(Octree &tree)
{
    const std::vector<Octant*>& treeVoxels = *(tree.voxel_);
    // 遍历每一帧Octree中的指定位置的voxel
    for (auto treeOctant : treeVoxels)
    {
        if (treeOctant->size == 0)          // TODO: 仅遍历每一帧Octree中包含点云的voxel位置
            continue;
        auto code = treeOctant->mortonCode; // 将指定位置取出 code标识Octant的位置
        auto baseOctant = createOctant(root_, code, 1, treeOctant->depth);      // 查找对应位置本树的Octant
        // 1、判断Octant的8个顶点是否都是固定Fixed
        bool isFixed = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (!baseOctant->isFixedLast[i])
            {
                isFixed = false;
                break;
            }
        }
        if (isFixed)    // 如果8个顶点都是固定Fixed 那么就排除该Octant
            continue;
        // 2、判断Octant的8个顶点是否存在固定Fixed
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (baseOctant->isFixedLast[i])         // 存在固定顶点那么将该Octant顶点及其与该顶点相邻的顶点都设置为非固定
            {
                baseOctant->isFixed[i] = false;
                for (uint32_t j = 0; j < 7; ++j)    // 遍历顶点相邻周围7个点
                {
                    auto neighborOctant = baseOctant->neighbor[cubeVertsNeighborTable[i][j]];
                    if (neighborOctant)
                        neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;
                }
            }
        }
    }  
}

/**
 * @description: 将当前帧的临时八叉树融合到主八叉树中 TSDF算法 同时进行MarchingCubes
 * @param {Octree} &tree 临时八叉树
 * @param {int} frames 帧序号
 * @return {*}
 */
void Octree::fusionVoxel(Octree &tree, int frames)
{
    const std::vector<Octant*>& treeVoxels = *(tree.voxel_);

    if (!params_.weightMode)        // 直出模式中要做融合前处理Fixed标识位 避免mesh之间出现缝隙
        voxelStatusCheck(tree);

    for (auto treeOctant : treeVoxels)
    {
        auto code = treeOctant->mortonCode;
        auto octant = createOctant(root_, code, 1, treeOctant->depth);
        // 是否使用allSampleMode
        if (params_.allSampleMode)
        {
            mcVoxels_->insert(octant);
        }

        // fusion 分为两部分 第一部分需要融合中心voxel(有点插入的)
        if (treeOctant->size != 0)              // 说明是中心voxel 我们直接更新其满足frames的8个点 采用加权更新
        {
            // 将当前帧点插入到树中
            fusionPoints(tree, octant, treeOctant);     // 按照voxel进行插入 每次插入会判断距离减少重复插入

            for (uint32_t i = 0; i < 8; ++i)    // 遍历8个顶点 判断是否满足更新条件
            {
                if (treeOctant->weight[i] == 0)    // 该点sdf无效或者被固定 跳过
                    continue;
                if (octant->frames[i] == -1)        // 从未被初始化
                {   
                    // 更新本身
                    octant->frames[i] = frames;
                    long weight = treeOctant->size;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            weight += neighborOctant->size;
                    }
                    octant->weight[i] = weight;                 // 更新weight
                    octant->sdf[i] = treeOctant->sdf[i];        // 更新sdf
                    octant->lastWeight[i] = weight;             // 更新lastWeight
                    // 更新周围节点(不存在就插入)
                    for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                    {
                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (!neighborOctant)                // 如果不存在
                        {
                            auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                            neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    // 创建节点
                        }
                        neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                        neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  // 更新weight
                        neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        // 更新sdf
                        neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        // 更新lastWeight
                    }
                }
                else if (octant->frames[i] != frames)   // 融合更新
                {
                    // 更新本身
                    octant->frames[i] = frames;
                    long weight = treeOctant->size;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            weight += neighborOctant->size;
                    }
                    // 判断是否可以更新
                    if (octant->isFixed[i])
                    {
                        if (params_.weightMode)     // 如果是加权模式 根据lastWeight和当前Weight比较判断是否加权SDF
                        {
                            if (weight - octant->lastWeight[i] < params_.reconTHR)
                            {
                                continue;
                            }
                            else
                            {
                                octant->isFixed[i] = false;
                                octant->isUpdated = false;
                                for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                                {
                                    auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                                    neighborOctant->isUpdated = false;
                                    neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;
                                }
                            }
                        }
                        else                        // 直出模式 Fixed位置不更新
                        {
                            continue;
                        }
                    }
                    if (!params_.weightMode)        // 直出模式非Fixed的位置要设置顶点及其周围顶点的isUpdated为false
                    {
                        octant->isUpdated = false;
                        for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                        {
                            auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                            neighborOctant->isUpdated = false;
                            neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;	 
                        }
                    }
                    double sdf = ((octant->weight[i] * octant->sdf[i]) + (weight * treeOctant->sdf[i])) / (octant->weight[i] + weight);
                    octant->weight[i] = octant->weight[i] + weight;     // 更新weight
                    octant->sdf[i] = sdf;                               // 更新sdf
                    octant->lastWeight[i] = weight;                     // 更新lastWeight
                    // 更新周围节点(不存在就插入)
                    for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                    {
                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (!neighborOctant)                // 如果不存在
                        {
                            auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                            neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    // 创建节点
                        }
                        neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                        neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  // 更新weight
                        neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        // 更新sdf
                        neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        // 更新lastWeight
                    }
                }
            }
        }
        // 第二部分需要融合边缘voxel(没点插入的)
        else
        {
            for (uint32_t i = 0; i < 8; ++i)    // 遍历8个顶点 判断是否满足更新条件
            {
                if (treeOctant->weight[i] == 0)    // 该点sdf无效或者被固定 跳过
                    continue;
                
                if (octant->frames[i] == -1)        // 从未被初始化
                {
                    // 更新本身
                    octant->frames[i] = frames;
                    long weight = treeOctant->size;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            weight += neighborOctant->size;
                    }
                    octant->weight[i] = weight;                 // 更新weight
                    octant->sdf[i] = treeOctant->sdf[i];        // 更新sdf
                    octant->lastWeight[i] = weight;             // 更新lastWeight
                    // 更新周围节点(不存在就插入)
                    for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                    {
                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (!neighborOctant)                // 如果不存在
                        {
                            auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                            neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    // 创建节点
                        }
                        neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                        neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  // 更新weight
                        neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        // 更新sdf
                        neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        // 更新lastWeight
                    }
                }
                else if (octant->frames[i] != frames)   // 融合更新
                {
                    // 判断该点周边是否有当前帧中心点
                    bool hasCenter = false;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            hasCenter = true;
                    }
                    if (hasCenter)          // 如果周边有中心点那么融合更新
                    {
                        // 更新本身
                        octant->frames[i] = frames;
                        long weight = treeOctant->size;
                        for (uint32_t j = 0; j < 7; ++j)
                        {
                            auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                            if (neighborOctant && neighborOctant->size > 0)
                                weight += neighborOctant->size;
                        }
                        // 判断是否可以更新
                        if (octant->isFixed[i])
                        {
                            if (params_.weightMode)      // 如果是加权模式 根据lastWeight和当前Weight比较判断是否加权SDF
                            {
                                if (weight - octant->lastWeight[i] < params_.reconTHR)
                                {
                                    continue;
                                }
                                else
                                {
                                    octant->isFixed[i] = false;
                                    octant->isUpdated = false;
                                    for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                                    {
                                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                                        neighborOctant->isUpdated = false;
                                        neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;
                                    }
                                }
                            }
                            else                        // 直出模式 Fixed位置不更新
                            {
                                continue;
                            }
                        }
                        if (!params_.weightMode)        // 直出模式非Fixed的位置要设置顶点及其周围顶点的isUpdated为false
                        {
                            octant->isUpdated = false;
                            for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                            {
                                auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                                neighborOctant->isUpdated = false;
                                neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false; 
                            }
                        }
                        double sdf = ((octant->weight[i] * octant->sdf[i]) + (weight * treeOctant->sdf[i])) / (octant->weight[i] + weight);
                        octant->weight[i] = octant->weight[i] + weight;     // 更新weight
                        octant->sdf[i] = sdf;                               // 更新sdf
                        octant->lastWeight[i] = weight;                     // 更新lastWeight
                        // 更新周围节点(不存在就插入)
                        for (uint32_t j = 0; j < 7; ++j)        // 遍历顶点周围7个点
                        {
                            auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                            if (!neighborOctant)                // 如果不存在
                            {
                                auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                                neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    // 创建节点
                            }
                            neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                            neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  // 更新weight
                            neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        // 更新sdf
                            neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        // 更新lastWeight
                        }
                    }
                }
            }
        }
    }
    // 执行MarchingCubes
    // 判断是否到达mc间隔
    if ((frames + 1) % params_.mcInterval)
        return;
    const std::vector<Octant*>& voxels = *voxel_;         // 叶子结点信息
    #pragma omp parallel for
    for (auto octant : voxels)
    {
        // 1、判断当前voxels是否可以mc 无效的voxel直接跳过
        bool isValid = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (octant->weight[i] == 0)
            {
                isValid = false;
                break;
            }
        }
        if (!isValid || octant->isUpdated)      // 跳过无效权重的voxel 以及 已经构建过mesh的voxel
            continue;
        // 2、根据分裂层级判断采用哪种subCube构建方式
        int level = params_.subLevel;      // 控制分裂层级
        Eigen::Tensor<double, 3> subSDF(level+1, level+1 , level+1);     // 定义保存SDF 大小和层级相关
        Eigen::Vector3i subIndex[level * level * level];                // 定义一个保存子坐标的数组 用于mc
        std::vector<bool> subMask(level * level * level, false);      // 定义一个mask用于确定是否需要mask掉没有点云的区域
        Eigen::Vector3d offsets(octant->x-(octant->extent/2), octant->y-(octant->extent/2), octant->z-(octant->extent/2));
        if (level > 1)                          // 当单个叶节点的分裂level大于1时 就要对voxel进行划分
        {
            SDFdeviation(octant, subIndex, subSDF, level);         // 曲率判断使用三线性差值或SDF重计算划分为多个subCube
        }
        else                                    // 当level=1普通方式 即一个标准正方体8个顶点
        {
            int num = 0;
            for (int a = 0; a <= 1; ++a) 
            {
                for (int b = 0; b <= 1; ++b) 
                {
                    for (int c = 0; c <= 1; ++c) 
                    {
                        if (a != 1 && b != 1 && c != 1)
                            subIndex[a * 1 * 1 + b * 1 + c] = Eigen::Vector3i(a, b, c); // 保存子cube的index序号
                        // 更新subSDF
                        subSDF(a, b, c) = octant->sdf[num++];
                    }
                }
            }
        }
        // 3、进行第一步MC 对每一个voxel进行
        #pragma omp critical
        {
            marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);         // 执行不带mask的MC subMask全为false
        }
        // 4、处理边缘voxel
        if (level > 1)
        {
            subMask.assign(level * level * level, true);        // 重新对subMask赋值 准备对voxel进行标记
            if (maskEdgeVoxel(octant, subMask, level))          // 对当前voxel进行判断 判断是否需要修剪voxel
            {
                #pragma omp critical
                {
                    trianglesClear(octant);                     // 重构voxel内的mesh前需要删除原有的mesh
                    marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);     // 执行带mask的MC
                }
            }
        }
        // 5、更新顶点及其相邻voxel顶点isFixed标识位 更新isFixedLast标识位（仅直出模式）
        octant->isUpdated = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            octant->isFixed[i] = true;
            if (!params_.weightMode)
                octant->isFixedLast[i] = true;
            for (uint32_t j = 0; j < 7; ++j)
            {
                auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = true;
                if (!params_.weightMode)
                    neighborOctant->isFixedLast[cubeVertsNeighborVTable[i][j]] = true;
            }
        }
        // 6、将当前octant插入到mcVoxels_中(不使用allSampleMode)
        if (!params_.allSampleMode)
        {
            #pragma omp critical
            {
                mcVoxels_->insert(octant);
                // octant->processFlags = true;
            }
        }
    }
}

/************************************ END ************************************/

/*********************************  其他函数  *********************************/
void Octree::updateVerts(torch::Tensor vertsOffset, torch::Tensor colorUpdate, torch::Tensor albedoUpdate, torch::Tensor shUpdate, torch::Tensor vertsIndex, torch::Tensor colorMask)
{
    VertsArray& verts = *verts_;
    auto offset = vertsOffset.accessor<float, 2>();
    auto index = vertsIndex.accessor<int, 2>();
    auto color = colorUpdate.accessor<float, 2>();
    auto albedo = albedoUpdate.accessor<float, 2>();
    auto sh = shUpdate.accessor<float, 2>();
    auto mask = colorMask.accessor<bool, 1>();
    for (uint32_t i = 0; i < offset.size(0); i++)
    {
        verts.updateOffset(index[i][0], Eigen::Vector3d(offset[i][0], offset[i][1], offset[i][2]), Eigen::Vector3d(color[i][0], color[i][1], color[i][2]), 
            Eigen::Vector3d(albedo[i][0], albedo[i][1], albedo[i][2]), Eigen::Vector4d(sh[i][0], sh[i][1], sh[i][2], sh[i][3]), mask[i]);
    }
    
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Octree::packVerts()
{
    VertsArray& verts = *verts_;
    std::vector<Eigen::Vector3i> triangles;     // 保存输出的

    const std::vector<Octant*>& voxels = *voxel_;         // 叶子结点信息 
    for (uint32_t i = 0; i < voxels.size(); ++i)           // 遍历叶节点
    {
        Octant* childOctant = voxels[i];

        if (childOctant->triangles)
            triangles.insert(triangles.end(), childOctant->triangles->begin(), childOctant->triangles->end());
    }
    torch::Tensor trianglesTensor = torch::zeros({static_cast<int64_t>(triangles.size()), 3}, torch::kInt);
    for (size_t i = 0; i < triangles.size(); ++i) {
        trianglesTensor[i][0] = triangles[i][0];
        trianglesTensor[i][1] = triangles[i][1];
        trianglesTensor[i][2] = triangles[i][2];
    }
    torch::Tensor vertsTensor, colorTensor, albedoTensor, shTensor;
    std::tie(vertsTensor, colorTensor, albedoTensor, shTensor) = verts.getVerts();

    return std::make_tuple(vertsTensor, trianglesTensor, colorTensor, albedoTensor, shTensor);
}

// 
/**
 * @description: 将mc生成的顶点坐标从octree中到处 生成verts顶点vector数组 以及标记三角面序号的triangles数组
 * @return {*}
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Octree::packOutput()
{
    const VertsArray& verts = *verts_;
    const std::unordered_set<Octant*>& voxels = *mcVoxels_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    const std::vector<bool>& normalConf = *normalConf_;               // 点云法向量置信度
    const std::vector<Eigen::Vector3d>& points = *data_;
    std::unordered_map<int, int> faceMap;                    // 用于存储face index映射
    std::unordered_set<Octant*> fliterVoxels;             // 经过过滤的voxels
    torch::Tensor trianglesTensor = torch::zeros({static_cast<int64_t>(voxels.size()*24), 3}, torch::kInt);     // 足够大的容量 避免超过内存
    torch::Tensor indexTensor = torch::zeros({static_cast<int64_t>(voxels.size()*48)}, torch::kInt);
    torch::Tensor vertsTensor = torch::zeros({static_cast<int64_t>(voxels.size()*48), 3}, torch::kFloat);
    torch::Tensor pointsTensor = torch::zeros({static_cast<int64_t>(points.size()), 3}, torch::kFloat);
    torch::Tensor normalsTensor = torch::zeros({static_cast<int64_t>(normals.size()), 3}, torch::kFloat);
    torch::Tensor trianglesMaskTensor = torch::zeros({static_cast<int64_t>(voxels.size()*24)}, torch::kBool);
    torch::Tensor vertsMaskTensor = torch::zeros({static_cast<int64_t>(voxels.size()*48)}, torch::kBool);
    int faceNum = 0;
    int vertsNum = 0;
    int pointsNum = 0;
    // std::unordered_set<Octant*> outsideNeighbors;
    // 遍历当前帧的voxels
    for (auto octant : voxels)
    {
        // 去除没有mesh的voxels
        if (octant->triangles == nullptr)
            continue;
        // 判断该voxel是否满足条件
        bool isMask = false;
        if (octant->curvature >= 0.01)
            isMask = true;
        
        // for (uint32_t i = 0; i < 26; ++i)
        // {
        //     auto neighborOctant = octant->neighbor[i];
        //     if (!neighborOctant)
        //         continue;
        //     // 在voxels查找是否存在neighborOctant
        //     if (neighborOctant->x == false)
        //     {
        //         outsideNeighbors.insert(neighborOctant);
        //     }
        //     else if (neighborOctant->curvature >= 0.01)
        //     {
        //         isMask = true;
        //     }
        // }
        // 遍历voxel中的face
        for (uint32_t i = 0; i<(*(octant->triangles)).size(); ++i)      
        {
            for (uint32_t j = 0; j < 3; ++j)
            {
                if (faceMap.find((*(octant->triangles))[i][j]) == faceMap.end())      // 非重复
                {
                    trianglesTensor[faceNum][j] = static_cast<int>(faceMap.size());
                    faceMap[(*(octant->triangles))[i][j]] = static_cast<int>(faceMap.size());
                    indexTensor[vertsNum] = ((*(octant->triangles))[i][j]);
                    vertsTensor[vertsNum][0] = verts[((*(octant->triangles))[i][j])].point[0];
                    vertsTensor[vertsNum][1] = verts[((*(octant->triangles))[i][j])].point[1];
                    vertsTensor[vertsNum][2] = verts[((*(octant->triangles))[i][j])].point[2];
                    // 添加顶点mask
                    if (isMask)
                    {
                        vertsMaskTensor[vertsNum] = true;
                    }
                    vertsNum++;
                }
                else
                {
                    trianglesTensor[faceNum][j] = faceMap[(*(octant->triangles))[i][j]];
                }

            }
            // 添加face mask
            if (isMask)
            {
                trianglesMaskTensor[faceNum] = true;
            }
            faceNum++;
        }
        if (isMask && (*(octant->triangles)).size() > 0)
        {
            // 遍历当前voxel中的点云 并保存到tensor中
            for (uint32_t i = 0; i < octant->size; ++i)
            {
                if (normalConf[octant->successors_[i]])
                {
                    pointsTensor[pointsNum][0] = points[octant->successors_[i]][0];
                    pointsTensor[pointsNum][1] = points[octant->successors_[i]][1];
                    pointsTensor[pointsNum][2] = points[octant->successors_[i]][2];
                    normalsTensor[pointsNum][0] = normals[octant->successors_[i]][0];
                    normalsTensor[pointsNum][1] = normals[octant->successors_[i]][1];
                    normalsTensor[pointsNum][2] = normals[octant->successors_[i]][2];
                    pointsNum++;
                }
            }
        }
    }
    // 添加周围邻居voxel
    // int testnum = 0;
    // for (auto octant : outsideNeighbors)
    // {
    //     if (octant->triangles == nullptr)
    //         continue;
    //     // testnum++;
    //     for (uint32_t i = 0; i<(*(octant->triangles)).size(); ++i)      
    //     {
    //         for (uint32_t j = 0; j < 3; ++j)
    //         {
    //             if (faceMap.find((*(octant->triangles))[i][j]) == faceMap.end())      // 非重复
    //             {
    //                 trianglesTensor[faceNum][j] = static_cast<int>(faceMap.size());
    //                 faceMap[(*(octant->triangles))[i][j]] = static_cast<int>(faceMap.size());
    //                 indexTensor[vertsNum] = ((*(octant->triangles))[i][j]);
    //                 vertsTensor[vertsNum][0] = verts[((*(octant->triangles))[i][j])].point[0];
    //                 vertsTensor[vertsNum][1] = verts[((*(octant->triangles))[i][j])].point[1];
    //                 vertsTensor[vertsNum][2] = verts[((*(octant->triangles))[i][j])].point[2];
    //                 vertsNum++;
    //             }
    //             else
    //             {
    //                 trianglesTensor[faceNum][j] = faceMap[(*(octant->triangles))[i][j]];
    //             }

    //         }
    //         // 添加face mask
    //         trianglesMaskTensor[faceNum] = true;
    //         faceNum++;
    //     }
    //     // 遍历当前voxel中的点云 并保存到tensor中
    //     for (uint32_t i = 0; i < octant->successors_.size(); ++i)
    //     {
    //         if (normalConf[octant->successors_[i]])
    //         {
    //             pointsTensor[pointsNum][0] = points[octant->successors_[i]][0];
    //             pointsTensor[pointsNum][1] = points[octant->successors_[i]][1];
    //             pointsTensor[pointsNum][2] = points[octant->successors_[i]][2];
    //             normalsTensor[pointsNum][0] = normals[octant->successors_[i]][0];
    //             normalsTensor[pointsNum][1] = normals[octant->successors_[i]][1];
    //             normalsTensor[pointsNum][2] = normals[octant->successors_[i]][2];
    //             pointsNum++;
    //         }
    //     }
    // }
    // std::cout << testnum << std::endl;

    // // 设置processFlags为False
    // for (auto octant : voxels)
    // {
    //     octant->processFlags = false;
    // }

    // 清空mcVoxels_
    mcVoxels_->clear();

    // 调整tensor的维度
    trianglesTensor = torch::narrow(trianglesTensor, 0, 0, faceNum);
    indexTensor = torch::narrow(indexTensor, 0, 0, vertsNum);
    vertsTensor = torch::narrow(vertsTensor, 0, 0, vertsNum);
    pointsTensor = torch::narrow(pointsTensor, 0, 0, pointsNum);
    normalsTensor = torch::narrow(normalsTensor, 0, 0, pointsNum);
    trianglesMaskTensor = torch::narrow(trianglesMaskTensor, 0, 0, faceNum);
    vertsMaskTensor = torch::narrow(vertsMaskTensor, 0, 0, vertsNum);

    return std::make_tuple(vertsTensor, trianglesTensor, indexTensor, pointsTensor, normalsTensor, vertsMaskTensor, trianglesMaskTensor);
}

std::tuple<torch::Tensor, torch::Tensor> Octree::packPointNoramls()
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    std::vector<bool>& normalConf = *normalConf_;               // 点云法向量置信度

    torch::Tensor pointsTensor = torch::zeros({static_cast<int64_t>(points.size()), 3}, torch::kFloat);
    torch::Tensor normalsTensor = torch::zeros({static_cast<int64_t>(normals.size()), 3}, torch::kFloat);

    int32_t pointIdx = 0;
    uint32_t validPointNum = 0;
    std::vector<uint32_t> resultIndices;
    std::vector<uint32_t> pointVec;
    subSuccessors(root_, pointVec);
    for (uint32_t i = 0; i< pointVec.size(); ++i)
    {
        pointIdx = pointVec[i];
        if (normalConf[pointIdx])
        {
            pointsTensor[pointIdx][0] = points[pointIdx][0];
            pointsTensor[pointIdx][1] = points[pointIdx][1];
            pointsTensor[pointIdx][2] = points[pointIdx][2];
            normalsTensor[pointIdx][0] = normals[pointIdx][0];
            normalsTensor[pointIdx][1] = normals[pointIdx][1];
            normalsTensor[pointIdx][2] = normals[pointIdx][2];
            validPointNum++;
        }
    }
    pointsTensor = torch::narrow(pointsTensor, 0, 0, validPointNum);
    normalsTensor = torch::narrow(normalsTensor, 0, 0, validPointNum);
    
    return std::make_tuple(pointsTensor, normalsTensor); 
}

/**
 * @description: 将SDF值打包为tensor输出到skimage库的marching cubes进行构建mash
 * @return {*}
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Octree::packCubeSDF()
{
    // 先获取octree的深度
    int64_t depth = 0;                  // 叶节点深度
    double extent = extent_;
    while(extent > params_.minExtent)
    {
        extent /= 2.0;
        depth++;
    }
    // 按照边长和深度构造一个cube
    int64_t level = params_.subLevel;           // 控制分裂层级
    int64_t csize = static_cast<int64_t>(std::pow(2, depth));
    torch::Tensor sdfCube = torch::zeros({(csize * level) + 1, (csize * level) + 1, (csize * level) + 1}, torch::kDouble);
    torch::Tensor maskCube = torch::zeros({(csize * level) + 1, (csize * level) + 1, (csize * level) + 1}, torch::kBool);
    torch::Tensor offsets = torch::zeros({4}, torch::kDouble);
    offsets[0] = center_[0] - (extent_ / 2);
    offsets[1] = center_[1] - (extent_ / 2);
    offsets[2] = center_[2] - (extent_ / 2);
    offsets[3] = extent_ / (csize * level);
    // 遍历所有的voxel插入到sdfCube中
    const std::vector<Octant*>& voxels = *voxel_;
    #pragma omp parallel for
    for (auto octant : voxels)
    {
        // 1、判断当前voxels是否可以mc 无效的voxel直接跳过
        bool isValid = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (octant->weight[i] == 0)
            {
                isValid = false;
                break;
            }
        }
        if (!isValid || octant->isUpdated)      // 跳过无效权重的voxel 以及 已经构建过mesh的voxel
            continue;
        // 2、根据分裂层级判断采用哪种subCube构建方式
        Eigen::Tensor<double, 3> subSDF(level+1, level+1 , level+1);     // 定义保存SDF 大小和层级相关
        Eigen::Vector3i subIndex[level * level * level];                // 定义一个保存子坐标的数组 用于mc
        std::vector<bool> subMask(level * level * level, false);      // 定义一个mask用于确定是否需要mask掉没有点云的区域
        Eigen::Vector3d offsets(octant->x-(octant->extent/2), octant->y-(octant->extent/2), octant->z-(octant->extent/2));
        if (level > 1)                          // 当单个叶节点的分裂level大于1时 就要对voxel进行划分
        {
            SDFdeviation(octant, subIndex, subSDF, level);         // 曲率判断使用三线性差值或SDF重计算划分为多个subCube
        }
        else                                    // 当level=1普通方式 即一个标准正方体8个顶点
        {
            int num = 0;
            for (int a = 0; a <= 1; ++a) 
            {
                for (int b = 0; b <= 1; ++b) 
                {
                    for (int c = 0; c <= 1; ++c) 
                    {
                        if (a != 1 && b != 1 && c != 1)
                            subIndex[a * 1 * 1 + b * 1 + c] = Eigen::Vector3i(a, b, c); // 保存子cube的index序号
                        // 更新subSDF
                        subSDF(a, b, c) = octant->sdf[num++];
                    }
                }
            }
        }
        // 3、进行第一步MC 对每一个voxel进行
        #pragma omp critical
        {
            marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);         // 执行不带mask的MC subMask全为false
        }
        // 4、处理边缘voxel
        if (level > 1)
        {
            subMask.assign(level * level * level, true);        // 重新对subMask赋值 准备对voxel进行标记
            if (maskEdgeVoxel(octant, subMask, level))          // 对当前voxel进行判断 判断是否需要修剪voxel
            {
                #pragma omp critical
                {
                    trianglesClear(octant);                     // 重构voxel内的mesh前需要删除原有的mesh
                    marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);     // 执行带mask的MC
                }
            }
            else                                                // 没有分裂区域mask为false
            {
                subMask.assign(level * level * level, false);
            }
        }
        // 5、更新顶点及其相邻voxel顶点isFixed标识位
        octant->isUpdated = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            octant->isFixed[i] = true;
            for (uint32_t j = 0; j < 7; ++j)
            {
                auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = true;
            }
        }
        // 6、将值写入数组
        auto code = octant->mortonCode;
        double extent = csize/2;
        Eigen::Vector3d index(csize/2, csize/2, csize/2);      // cube的原点坐标(相对)
        for (int i = depth-1; i >= 0; --i)
        {
            auto tmpCode = (code >> (i*3)) & 0x07;   // 获取morton code
            int bit2 = (tmpCode >> 2) & 1; // 获取第三位
            int bit1 = (tmpCode >> 1) & 1; // 获取第二位
            int bit0 = tmpCode & 1;       // 获取第一位
            extent /= 2;
            // 更新坐标
            if (bit0)
                index[0] = index[0] + extent;
            else
                index[0] = index[0] - extent;
            if (bit1)
                index[1] = index[1] + extent;
            else
                index[1] = index[1] - extent;
            if (bit2)
                index[2] = index[2] + extent;
            else
                index[2] = index[2] - extent;
        }
        index[0] -= 0.5; index[1] -= 0.5; index[2] -= 0.5;   // 放置在原点上(相对)
        index[0] *= level; index[1] *= level; index[2] *= level;

        for (int a = 0; a <= level; ++a) {
            for (int b = 0; b <= level; ++b) {
                for (int c = 0; c <= level; ++c) {
                    sdfCube[index[0] + a][index[1] + b][index[2] + c] = subSDF(a, b, c);        // 写入SDF数组
                    if (a < level && b < level && c < level)
                    {
                        if (!subMask[a * level * level + b * level + c])
                            maskCube[index[0] + a + 1][index[1] + b + 1][index[2] + c + 1] = true;
                    }
                    
                }
            }
        }
    }

    return std::make_tuple(sdfCube, maskCube, offsets);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Octree::packSDF()
{
    const std::vector<Octant*>& voxels = *voxel_;
    torch::Tensor voxelTensor = torch::zeros({static_cast<int64_t>(voxels.size()), 8, 3}, torch::kDouble);  // 创建大小为(N,8,3)的Tensor
    torch::Tensor sdfTensor = torch::zeros({static_cast<int64_t>(voxels.size()), 8}, torch::kDouble);       // 创建大小为(N,8)的Tensor
    torch::Tensor codeTensor = torch::zeros({static_cast<int64_t>(voxels.size())}, torch::kLong);         // 创建大小为(N,1)的Tensor
    uint32_t i = 0;
    for (auto octant : voxels)    // 遍历所有voxel 对于有sdf值的voxel进行输出
    {
        if (octant->sdf == nullptr)
            continue;

        // 计算叶节点边长和深度
        double extent = extent_;         // octree最大边长
        while(extent > params_.minExtent)
        {
            extent /= 2.0;
        }

        for (int a = 0; a < 2; ++a)         // x方向
        {
            double x = octant->x + (a % 2 == 0 ? -1 : 1) * extent / 2.0;
            for (int b = 0; b < 2; ++b)     // y方向
            {
                double y = octant->y + (b % 2 == 0 ? -1 : 1) * extent / 2.0;
                for (int c = 0; c < 2; ++c) // z方向
                {
                    double z = octant->z + (c % 2 == 0 ? -1 : 1) * extent / 2.0;
                    voxelTensor[i][4*a+2*b+c][0] = x;
                    voxelTensor[i][4*a+2*b+c][1] = y;
                    voxelTensor[i][4*a+2*b+c][2] = z;
                }
            }
        }
        auto sdf = torch::from_blob(octant->sdf, {8}, dtype(torch::kDouble));
        sdfTensor[i] = sdf;
        codeTensor[i] = static_cast<long>(octant->mortonCode);
        i++;
    }
    voxelTensor.resize_({i+1, 8, 3});
    sdfTensor.resize_({i+1, 8});
    codeTensor.resize_({i+1});

    return std::make_tuple(voxelTensor, sdfTensor, codeTensor);
}

/**
 * @description: 接口函数 输入深度 内参和当前帧Rt 进行树插入及其融合操作
 * @param {Tensor} depth 深度图
 * @param {Tensor} K 相机内参
 * @param {Tensor} Rt 当前帧位姿
 * @return {*}
 */
void Octree::updateTree(const torch::Tensor depth, const torch::Tensor K, const torch::Tensor Rt, int64_t frames)
{
    // 将深度通过Rt投影到空间中
    auto pointsT = transformToPointCloud(depth, K, Rt);
    // 转换为Eigen
    auto points = libtorch2eigen<float>(pointsT);
    // 取出Rt的平移分量
    // auto T = Rt.slice(1, 3, 4).slice(0, 0, 3).squeeze(1);   
    auto cameraRt = Rt.accessor<float, 2>();
    Eigen::Vector3d camera(cameraRt[0][3], cameraRt[1][3], cameraRt[2][3]);
    // 对每一帧都要构造一个临时的octree用于保存节点信息
    Octree frameTree(center_, extent_, params_);     // 构造一个临时的Octree
    frameTree.insert(points.cast<double>(), camera);    // 插入点云
    // 融合到主Octree中
    fusionVoxel(frameTree, frames);
}

/**
 * @description: 接口函数 输入点云 内参和当前帧Rt 进行树插入及其融合操作
 * @param {Tensor} points 点云相机坐标系
 * @param {Tensor} K 相机内参
 * @param {Tensor} Rt 当前帧位姿
 * @return {*}
 */
void Octree::updateTreePcd(const torch::Tensor pointData, const torch::Tensor K, const torch::Tensor Rt, int64_t frames)
{
    auto camera_coords = torch::cat({pointData, torch::ones({pointData.size(0), 1})}, 1);
    auto camera_coordsT = camera_coords.transpose(1, 0);
    torch::Tensor world_coords = Rt.mm(camera_coordsT);
    torch::Tensor pointWorld = world_coords.t().slice(1, 0, 3);
    // 转换为Eigen
    auto points = libtorch2eigen<float>(pointWorld);
    // 取出Rt的平移分量
    // auto T = Rt.slice(1, 3, 4).slice(0, 0, 3).squeeze(1);   
    auto cameraRt = Rt.accessor<float, 2>();
    Eigen::Vector3d camera(cameraRt[0][3], cameraRt[1][3], cameraRt[2][3]);
    // 对每一帧都要构造一个临时的octree用于保存节点信息
    Octree frameTree(center_, extent_, params_);     // 构造一个临时的Octree
    frameTree.insert(points.cast<double>(), camera);    // 插入点云
    // 融合到主Octree中
    fusionVoxel(frameTree, frames);
}

/************************************ END ************************************/