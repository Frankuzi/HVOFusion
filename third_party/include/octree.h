#pragma once
#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <torch/script.h>
#include <torch/custom_class.h>

// 构造一个保存点坐标和序号列表的结构体
struct PointWithList 
{
    Eigen::Vector3d point;
    Eigen::Vector3d color;
    Eigen::Vector3d albedo;
    Eigen::Vector4d sh;
    Eigen::Vector3d offset = Eigen::Vector3d(0, 0, 0);     // mesh顶点的偏移
    std::vector<uint32_t> linkedList;
    bool isValid;       // 表示这个位置是否有效
    PointWithList(Eigen::Vector3d pt, uint32_t id) : point(pt), isValid(true)    // 结构体初始化函数
    {
        linkedList.push_back(id);
    }
};

// 构造一个保存顶点的数组
class VertsArray {
public:
    VertsArray() {}

    // 插入模式1 raw_insert 全新插入 返回插入的index index对应std::vector<PointWithList> values_中的序号
    uint32_t raw_insert(Eigen::Vector3d pt, uint32_t id) {
        // 判断是否有空位置
        int index;
        if (!gaps_.empty()) {
            index = *gaps_.begin();         // 如果整个列表有空值
            gaps_.erase(index);
            values_[index].point = pt;
            values_[index].offset = Eigen::Vector3d(0, 0, 0);
            values_[index].linkedList.clear();
            values_[index].linkedList.push_back(id);
            values_[index].isValid = true;
        }
        else
        {
            values_.push_back(PointWithList(pt, id));
            index = values_.size() - 1;
        }
        return index;
    }

    // 插入模式2 id_update 在已有的元素位置上插入新的序号
    bool id_update(uint32_t index, Eigen::Vector3d pt, uint32_t id)
    {
        if (index >= values_.size())
            return false;
        if (!values_[index].isValid)
            return false;

        // values_[index].point = pt;
        values_[index].linkedList.push_back(id);
        return true;
    }

    // 删除指定index的元素
    bool remove(uint32_t index, uint32_t id)
    {
        if (index >= values_.size() || !(values_[index].isValid)) 
            return false;
        auto it = std::find(values_[index].linkedList.begin(), values_[index].linkedList.end(), id);
        if (it == values_[index].linkedList.end())
            return false;

        if (values_[index].linkedList.size() > 1)
        {
            values_[index].linkedList.erase(it);
        }
        else
        {
            values_[index].linkedList.clear();
            // 删除值，将位置标记为空
            values_[index].isValid = false;
            // 添加到空缺集合中
            gaps_.insert(index);
        }

        return true;
    }

    bool updateOffset(uint32_t index, Eigen::Vector3d offset, Eigen::Vector3d color, Eigen::Vector3d albedo, Eigen::Vector4d sh, bool mask)
    {
        if (index >= values_.size() || !(values_[index].isValid)) 
            return false;
        values_[index].offset = offset;
        if (mask)
        {
            values_[index].color = color;
            values_[index].albedo = albedo;
            values_[index].sh = sh;
        }
        return true;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getVerts()
    {
        torch::Tensor vertsTensor = torch::zeros({static_cast<int64_t>(values_.size()), 3}, torch::kDouble);     // 根据点云数量新建一个torch Tensor
        torch::Tensor colorTensor = torch::zeros({static_cast<int64_t>(values_.size()), 3}, torch::kDouble);
        torch::Tensor albedoTensor = torch::zeros({static_cast<int64_t>(values_.size()), 3}, torch::kDouble);
        torch::Tensor shTensor = torch::zeros({static_cast<int64_t>(values_.size()), 4}, torch::kDouble);
        for (size_t i = 0; i < values_.size(); ++i) {
            vertsTensor[i][0] = values_[i].point[0] + values_[i].offset[0];
            vertsTensor[i][1] = values_[i].point[1] + values_[i].offset[1];
            vertsTensor[i][2] = values_[i].point[2] + values_[i].offset[2];
            colorTensor[i][0] = values_[i].color[0];
            colorTensor[i][1] = values_[i].color[1];
            colorTensor[i][2] = values_[i].color[2];
            albedoTensor[i][0] = values_[i].albedo[0];
            albedoTensor[i][1] = values_[i].albedo[1];
            albedoTensor[i][2] = values_[i].albedo[2];
            shTensor[i][0] = values_[i].sh[0];
            shTensor[i][1] = values_[i].sh[1];
            shTensor[i][2] = values_[i].sh[2];
            shTensor[i][3] = values_[i].sh[3];
        }
        
        return std::make_tuple(vertsTensor, colorTensor, albedoTensor, shTensor);
    }

    std::vector<Eigen::Vector3d> getBorder()
    {
        std::vector<Eigen::Vector3d> points;
        for (uint32_t i = 0; i < values_.size(); ++i)
        {
            if (values_[i].linkedList.size() <= 3)
            {
                points.push_back(values_[i].point);
            }
        }
        return points;
    }

    PointWithList& operator[](uint32_t index) 
    {
        if (index >= values_.size() || !(values_[index].isValid)) {
            throw std::out_of_range("Invalid index");
        }
        return values_[index];
    }

    const PointWithList& operator[](uint32_t index) const 
    {
        if (index >= values_.size() || !(values_[index].isValid)) {
            throw std::out_of_range("Invalid index");
        }
        return values_[index];
    }

private:
    std::vector<PointWithList> values_;
    std::unordered_set<uint32_t> gaps_;      // 保存空缺位置
};


// 对于两个不同空间点进行测距
struct L1Distance
{
    static inline double compute(const Eigen::Vector3d &p, const Eigen::Vector3d &q)
    {
        double diff1 = p[0] - q[0];
        double diff2 = p[1] - q[1];
        double diff3 = p[2] - q[2];

        return std::abs(diff1) + std::abs(diff2) + std::abs(diff3);
    }

    static inline double norm(double x, double y, double z)
    {
        return std::abs(x) + std::abs(y) + std::abs(z);
    }

    static inline double sqr(double r)
    {
        return r;
    }

    static inline double sqrt(double r)
    {
        return r;
    }
};

struct L2Distance
{
    static inline double compute(const Eigen::Vector3d &p, const Eigen::Vector3d &q)
    {
        double diff1 = p[0] - q[0];
        double diff2 = p[1] - q[1];
        double diff3 = p[2] - q[2];

        return std::pow(diff1, 2) + std::pow(diff2, 2) + std::pow(diff3, 2);
    }

    static inline double norm(double x, double y, double z)
    {
        return std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2);
    }

    static inline double sqr(double r)
    {
        return r * r;
    }

    static inline double sqrt(double r)
    {
        return std::sqrt(r);
    }
};

// 八叉树参数配置
struct OctreeParams
{
    float minExtent;        // 最小的voxel边长
    float minSize;          // 点云点之间的间隔
    uint32_t pointsValid;   // 点云中有效点的邻居点判断阈值
    float normalRadius;     // 计算法向量的搜索半径
    float sdfRadius;        // 计算SDF的搜索半径
    float curvatureTHR;     // 重计算SDF的曲率判定阈值 对voxel分级subLevel时是否需要重计算SDF
    uint32_t reconTHR;      // 用于判断该Fixed的Voxel是否需要更新（加权方法中 如果某个voxel后一帧点云数量大于前一帧点云超过reconTHR阈值则需要更新）
    float minBorder;        // 用于判断当前voxel生成的mesh是否需要裁剪
    uint32_t mcInterval;    // 执行MarchingCubes的间隔 和输出间隔保持一致
    int subLevel;           // 每个叶子结点的voxel分裂层级
    bool weightMode;        // 构建mesh的方式：false——SDF直出不加权，true——SDF加权累计满mcInterval后再输出
    bool allSampleMode;     // 每一帧输出的voxel是否输出所有sample
    bool meshOverlap = false;   // 帧之间的mesh是否可以重叠生成
    // IPHONE: 
    // float minExtent = 0.02f, float minSize = 0.001, int pointsValid = 25, float normalRadius = 0.025, float sdfRadius = 0.02,
    // float curvatureTHR = 0.01, uint32_t reconTHR = 50, float minBorder = 0.022, uint32_t mcInterval = 10, int subLevel = 1
    // REPLICA: 
    // float minExtent = 0.1f, float minSize = 0.014, int pointsValid = 40, float normalRadius = 0.1, float sdfRadius = 0.12,
    // float curvatureTHR = 0.01, uint32_t reconTHR = 50, float minBorder = 0.022, uint32_t mcInterval = 10, int subLevel = 1
public:
    OctreeParams(float minExtent = 0.05f, float minSize = 0.014, int pointsValid = 5, float normalRadius = 0.05, float curvatureTHR = 0.01,
                float sdfRadius = 0.5, uint32_t reconTHR = 50, float minBorder = 0.022, uint32_t mcInterval = 20, int subLevel = 1,
                bool weightMode = true, bool allSampleMode = true)
        : minExtent(minExtent), minSize(minSize), pointsValid(pointsValid), normalRadius(normalRadius), sdfRadius(sdfRadius),
        curvatureTHR(curvatureTHR), reconTHR(reconTHR), minBorder(minBorder), mcInterval(mcInterval), subLevel(subLevel),
        weightMode(weightMode), allSampleMode(allSampleMode)
    {
    }
};

// 八叉树类
class Octree : public torch::CustomClassHolder // torch::CustomClassHolder 将这个类构建成Python接口
{
public:
    Octree();

    Octree(const Eigen::Vector3d center, double extent, OctreeParams params);

    ~Octree();

    void init(const torch::Tensor center, double extent, double minExtent, double minSize, int64_t pointsValid, double normalRadius,
        double curvatureTHR, double sdfRadius, int64_t reconTHR, double minBorder, int64_t mcInterval, int64_t subLevel, bool weightMode, bool allSampleMode);

    void clear();

    void updateTree(const torch::Tensor depth, const torch::Tensor K, const torch::Tensor Rt, int64_t frames);

    void updateTreePcd(const torch::Tensor pointData, const torch::Tensor K, const torch::Tensor Rt, int64_t frames);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> packOutput();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> packVerts();

    std::tuple<torch::Tensor, torch::Tensor> packPointNoramls();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> packSDF();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> packCubeSDF();

    void updateVerts(torch::Tensor offset, torch::Tensor color, torch::Tensor albedo, torch::Tensor sh, torch::Tensor index, torch::Tensor mask);

protected:
    class Octant
    {
    public:
        Octant();
        ~Octant();

        bool isLeaf;                         // 是否为叶节点
        bool *isFixed;                       // 表示8个顶点的SDF是否可以更新 (仅对叶子节点有效)
        bool *isFixedLast;                   // 保存上一帧Fixed的Voxel（仅在weightMode=false使用）
        bool isUpdated;                      // 表示该voxel是否需要mc (仅对叶子节点有效)
        double x, y, z;                      // 节点中心坐标
        double extent;                       // side-length
        uint32_t depth;                      // depth
        double curvature;                    // 该节点中点云的平均曲率 (仅对叶子节点有效)
        double *sdf;                         // 定义一个sdf指针 用于保存当前voxel的8个顶点sdf值 (仅对叶子节点有效)
        std::vector<Eigen::Vector3i> *triangles; // 定义一个指针 指向保存当前mesh的face面序号数组 (仅对叶子节点有效)
        Octant *child[8];                    // 8个子节点指针
        Octant *neighbor[26];                // 26个周围节点指针 (仅对叶子节点有效)
        uint32_t mortonCode;                 // 节点的mortoncode编码
        uint32_t size;                       // 包含子节点数量
        int *frames;                         // 该节点的8个顶点融合到第几帧 (仅对叶子节点有效)
        // bool processFlags;                   // 是否为当前处理的voxels 和mcVoxels保持一致 (仅对叶子节点有效)
        uint32_t *weight;                    // sdf融合权重 对于每个临时帧的octree它用于表示该点是否有效 (仅对叶子节点有效)
        uint32_t *lastWeight;                // 用于当前Fixed点是否需要更新的判断条件 (仅对叶子节点有效)
        uint32_t curveWeight;                // 用于记录当前曲率融合权重 (仅对叶子节点有效)
        std::vector<uint32_t> successors_;   // 每个叶节点都带一个保存点云index的链表 (仅对叶子节点有效)
    };

    // not copyable, not assignable ...
    Octree(Octree &);
    Octree &operator=(const Octree &oct);

    void radiusNeighbors(const Octant *octant, const Eigen::Vector3d &query, double radius,
                         double sqrRadius, std::vector<uint32_t> &resultIndices) const;

    bool findNeighbor(const Octant *octant, const Eigen::Vector3d &query, double minDistance,
                      double &maxDistance, uint32_t &resultIndex) const;

    void insert(const Eigen::Matrix<double, Eigen::Dynamic, 3> &points, const Eigen::Vector3d &camera);
    
    void insertOctant(const Eigen::Vector3d& point, Octant* octant, double extent, uint32_t morton, std::unordered_set<Octant*> &voxels);

    Octant* createOctant(Octant* octant, uint32_t morton, uint32_t startDepth, const uint32_t depth);
    
    Octant* createOctantSimply(Octant* octant, uint32_t morton, uint32_t startDepth, const uint32_t depth);

    static bool contains(const Eigen::Vector3d &query, double sqRadius, const Octant *octant);

    static bool overlaps(const Eigen::Vector3d &query, double radius, double sqRadius, const Octant *o);

    static bool inside(const Eigen::Vector3d &query, double radius, const Octant *octant);

    void updateNormals(const Eigen::Vector3d &camera, std::unordered_set<Octant*> &voxels);

    void fusionVoxel(Octree &tree, int frames);
    
    void voxelStatusCheck(Octree &tree);

    void updateVoxel(std::unordered_set<Octant*> &voxels);
    
    void updateCurvature(Octant *octant);

    void fusionPoints(Octree &tree, Octant *baseOctant, Octant *treeOctant);

    bool maskEdgeVoxel(Octant* octant, std::vector<bool> &subMask, int level);

    void SDFdeviation(const Octant* octant, Eigen::Vector3i *subIndex, Eigen::Tensor<double, 3> &subSDF, int level);
    
    static void subSuccessors(const Octant* octant, std::vector<uint32_t> &successors);

    void marchingCubesSparse(Octant* octant, const Eigen::Vector3i* valid_cords,
                            const Eigen::Tensor<double, 3> &dense_sdf, const std::vector<bool> mask, const uint32_t num_lif,
                            const Eigen::Vector3d offsets);

    void trianglesClear(Octant* octant);

    OctreeParams params_;
    Octant *root_;                         // 根节点的指针
    double extent_;                         // 节八叉树边长大小
    Eigen::Vector3d center_;               // 八叉树中心点坐标
    std::vector<Eigen::Vector3d> *data_;   // 保存点云数据
    std::vector<Eigen::Vector3d> *normal_; // 保存法向量
    std::vector<bool> *normalConf_;        // 保存法向量置信度
    std::vector<double> *curvature_;         // 保存点云曲率
    VertsArray *verts_;                    // 保存Marching Cubes后的顶点
    std::vector<Octant*> *voxel_;          // 保存叶节点节点
    std::unordered_set<Octant*> *mcVoxels_;       // 保存当前帧mc的voxels
};

// 节点相关性表
// 均右手系坐标系方向
// 寻找每个正方体的相邻正方体（有一条边重合都算相邻）一共18个 这里的3保存了相邻正方体距离中心正方体的偏移位置 增加另外8个顶点位置
const int neighborTable[26][3] = {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {-1, 1, 0}, {-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}, {1, 0, 1}, {0, 1, 1}, {-1, 0, 1}, {0, -1, 1}, {0, 0, 1}, {1, 0, -1}, {0, 1, -1}, {-1, 0, -1}, {0, -1, -1}, {0, 0, -1}, {1, 1, 1}, {1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};
const int oppNeighborTableID[26] = {4, 5, 6, 7, 0, 1, 2, 3, 15, 16, 13, 14, 17, 10, 11, 8, 9, 12, 24, 25, 22, 23, 20, 21, 18, 19};
// 每个cube有26个邻居 这里记录了26个邻居受到影响的顶点在updateVoxel函数中cubes的序号（记录了邻居顶点受影响的位置）
const int oppOverlapTable[26][8] = {{1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0},
                                    {0, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 0, 0, 0},
                                    {1, 0, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 0, 0, 1, 0},
                                    {1, 0, 1, 0, 1, 0, 1, 0}, {0, 1, 0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 1},
                                    {0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 1, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1, 0, 0}};
// 每个cube有26个邻居 这里记录了26个邻居受到影响的顶点在updateVoxel函数中cubes的序号（与上述oppOverlapTable对应，邻居节点影响的位置在中心节点的位置）
const int selfOverlapTable[26][8] = {{4, 5, 6, 7, -1, -1, -1, -1}, {6, 7, -1, -1, -1, -1, -1, -1}, {2, 3, -1, -1, 6, 7, -1, -1}, {-1, -1, -1, -1, 2, 3, -1, -1},
                                    {-1, -1, -1, -1, 0, 1, 2, 3}, {-1, -1, -1, -1, -1, -1, 0, 1}, {-1, -1, 0, 1, -1, -1, 4, 5}, {-1, -1, 4, 5, -1, -1, -1, -1},
                                    {5, -1, 7, -1, -1, -1, -1, -1}, {3, -1, -1, -1, 7, -1, -1, -1}, {-1, -1, -1, -1, 1, -1, 3, -1}, {-1, -1, 1, -1, -1, -1, 5, -1},
                                    {1, -1, 3, -1, 5, -1, 7, -1}, {-1, 4, -1, 6, -1, -1, -1, -1}, {-1, 2, -1, -1, -1, 6, -1, -1}, {-1, -1, -1, -1, -1, 0, -1, 2},
                                    {-1, -1, -1, 0, -1, -1, -1, 4}, {-1, 0, -1, 2, -1, 4, -1, 6}, {7, -1, -1, -1, -1, -1, -1, -1}, {-1, -1, 5, -1, -1, -1, -1, -1},
                                    {-1, -1, -1, -1, -1, -1, 1, -1}, {-1, -1, -1, -1, 3, -1, -1, -1}, {-1, 6, -1, -1, -1, -1, -1, -1}, {-1, -1, -1, 4, -1, -1, -1, -1},
                                    {-1, -1, -1, -1, -1, -1, -1, 0}, {-1, -1, -1, -1, -1, 2, -1, -1}};
// cube的每一个顶点共享的voxel 与neighborTable序号对应
const int cubeVertsNeighborTable[8][7] = {{6, 5, 4, 16, 24, 15, 17}, {6, 5, 4, 11, 20, 10, 12}, {4, 3, 2, 15, 25, 14, 17}, {4, 3, 2, 10, 21, 9, 12}, {0, 7, 6, 13, 23, 16, 17}, {0, 7, 6, 8, 19, 11, 12}, {0, 1, 2, 13, 22, 14, 17}, {0, 1, 2, 8, 18, 9, 12}};
const int cubeVertsNeighborVTable[8][7] = {{2, 6, 4, 3, 7, 5, 1}, {3, 7, 5, 2, 6, 4, 0}, {6, 4, 0, 7, 5, 1, 3}, {7, 5, 1, 6, 4, 0, 2}, {0, 2, 6, 1, 3, 7, 5}, {1, 3, 7, 0, 2, 6, 4}, {2, 0, 4, 3, 1, 5, 7}, {3, 1, 5, 2, 0, 4, 6}};