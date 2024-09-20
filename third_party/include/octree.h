#pragma once
#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <torch/script.h>
#include <torch/custom_class.h>

struct PointWithList 
{
    Eigen::Vector3d point;
    Eigen::Vector3d color;
    Eigen::Vector3d albedo;
    Eigen::Vector4d sh;
    Eigen::Vector3d offset = Eigen::Vector3d(0, 0, 0);     
    std::vector<uint32_t> linkedList;
    bool isValid;       
    PointWithList(Eigen::Vector3d pt, uint32_t id) : point(pt), isValid(true)    
    {
        linkedList.push_back(id);
    }
};

class VertsArray {
public:
    VertsArray() {}

    uint32_t raw_insert(Eigen::Vector3d pt, uint32_t id) {
        int index;
        if (!gaps_.empty()) {
            index = *gaps_.begin();         
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
            values_[index].isValid = false;
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
        torch::Tensor vertsTensor = torch::zeros({static_cast<int64_t>(values_.size()), 3}, torch::kDouble);     
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
    std::unordered_set<uint32_t> gaps_;      
};


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

struct OctreeParams
{
    float minExtent;        
    float minSize;          
    uint32_t pointsValid;   
    float normalRadius;     
    float sdfRadius;        
    float curvatureTHR;    
    uint32_t reconTHR;      
    float minBorder;        
    uint32_t mcInterval;    
    int subLevel;           
    bool weightMode;        
    bool allSampleMode;     
    bool meshOverlap = false;   
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

class Octree : public torch::CustomClassHolder 
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

        bool isLeaf;                         
        bool *isFixed;                       
        bool *isFixedLast;                   
        bool isUpdated;                      
        double x, y, z;                      
        double extent;                       
        uint32_t depth;                      
        double curvature;                    
        double *sdf;                         
        std::vector<Eigen::Vector3i> *triangles; 
        Octant *child[8];                    
        Octant *neighbor[26];                
        uint32_t mortonCode;                 
        uint32_t size;                       
        int *frames;                         
        uint32_t *weight;                    
        uint32_t *lastWeight;                
        uint32_t curveWeight;                
        std::vector<uint32_t> successors_;   
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
    Octant *root_;                         
    double extent_;                         
    Eigen::Vector3d center_;               
    std::vector<Eigen::Vector3d> *data_;   
    std::vector<Eigen::Vector3d> *normal_; 
    std::vector<bool> *normalConf_;        
    std::vector<double> *curvature_;         
    VertsArray *verts_;                    
    std::vector<Octant*> *voxel_;          
    std::unordered_set<Octant*> *mcVoxels_;       
};

const int neighborTable[26][3] = {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {-1, 1, 0}, {-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}, {1, 0, 1}, {0, 1, 1}, {-1, 0, 1}, {0, -1, 1}, {0, 0, 1}, {1, 0, -1}, {0, 1, -1}, {-1, 0, -1}, {0, -1, -1}, {0, 0, -1}, {1, 1, 1}, {1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}};
const int oppNeighborTableID[26] = {4, 5, 6, 7, 0, 1, 2, 3, 15, 16, 13, 14, 17, 10, 11, 8, 9, 12, 24, 25, 22, 23, 20, 21, 18, 19};
const int oppOverlapTable[26][8] = {{1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0},
                                    {0, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 0, 0, 0},
                                    {1, 0, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 0, 0, 1, 0},
                                    {1, 0, 1, 0, 1, 0, 1, 0}, {0, 1, 0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 1},
                                    {0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 1, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1, 0, 0}};
const int selfOverlapTable[26][8] = {{4, 5, 6, 7, -1, -1, -1, -1}, {6, 7, -1, -1, -1, -1, -1, -1}, {2, 3, -1, -1, 6, 7, -1, -1}, {-1, -1, -1, -1, 2, 3, -1, -1},
                                    {-1, -1, -1, -1, 0, 1, 2, 3}, {-1, -1, -1, -1, -1, -1, 0, 1}, {-1, -1, 0, 1, -1, -1, 4, 5}, {-1, -1, 4, 5, -1, -1, -1, -1},
                                    {5, -1, 7, -1, -1, -1, -1, -1}, {3, -1, -1, -1, 7, -1, -1, -1}, {-1, -1, -1, -1, 1, -1, 3, -1}, {-1, -1, 1, -1, -1, -1, 5, -1},
                                    {1, -1, 3, -1, 5, -1, 7, -1}, {-1, 4, -1, 6, -1, -1, -1, -1}, {-1, 2, -1, -1, -1, 6, -1, -1}, {-1, -1, -1, -1, -1, 0, -1, 2},
                                    {-1, -1, -1, 0, -1, -1, -1, 4}, {-1, 0, -1, 2, -1, 4, -1, 6}, {7, -1, -1, -1, -1, -1, -1, -1}, {-1, -1, 5, -1, -1, -1, -1, -1},
                                    {-1, -1, -1, -1, -1, -1, 1, -1}, {-1, -1, -1, -1, 3, -1, -1, -1}, {-1, 6, -1, -1, -1, -1, -1, -1}, {-1, -1, -1, 4, -1, -1, -1, -1},
                                    {-1, -1, -1, -1, -1, -1, -1, 0}, {-1, -1, -1, -1, -1, 2, -1, -1}};
const int cubeVertsNeighborTable[8][7] = {{6, 5, 4, 16, 24, 15, 17}, {6, 5, 4, 11, 20, 10, 12}, {4, 3, 2, 15, 25, 14, 17}, {4, 3, 2, 10, 21, 9, 12}, {0, 7, 6, 13, 23, 16, 17}, {0, 7, 6, 8, 19, 11, 12}, {0, 1, 2, 13, 22, 14, 17}, {0, 1, 2, 8, 18, 9, 12}};
const int cubeVertsNeighborVTable[8][7] = {{2, 6, 4, 3, 7, 5, 1}, {3, 7, 5, 2, 6, 4, 0}, {6, 4, 0, 7, 5, 1, 3}, {7, 5, 1, 6, 4, 0, 2}, {0, 2, 6, 1, 3, 7, 5}, {1, 3, 7, 0, 2, 6, 4}, {2, 0, 4, 3, 1, 5, 7}, {3, 1, 5, 2, 0, 4, 6}};
