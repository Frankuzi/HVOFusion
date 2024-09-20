#include <omp.h>
#include "../include/octree.h"
#include "../include/normals.hpp"
#include "../include/utils.h"


Octree::Octree()
    : root_(0), extent_(0), data_(0), normal_(0), normalConf_(0), curvature_(0), verts_(0), voxel_(0), mcVoxels_(0)
{
}

Octree::Octree(const Eigen::Vector3d center, double extent, OctreeParams params)
{
    params_ = params;
    center_[0] = center[0];
    center_[1] = center[1];
    center_[2] = center[2];
    root_ = new Octant;
    root_->x = center[0];
    root_->y = center[1];
    root_->z = center[2];
    root_->extent = extent;    
    root_->depth = 0;           
    extent_ = extent;
    data_ = new std::vector<Eigen::Vector3d>;
    normal_ = new std::vector<Eigen::Vector3d>;
    normalConf_ = new std::vector<bool>;
    curvature_ = new std::vector<double>;
    verts_ = new VertsArray;
    voxel_ = new std::vector<Octant*>;
    mcVoxels_ = new std::unordered_set<Octant*>;
}

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

void Octree::init(const torch::Tensor center, double extent, double minExtent, double minSize, int64_t pointsValid, double normalRadius,
        double curvatureTHR, double sdfRadius, int64_t reconTHR, double minBorder, int64_t mcInterval, int64_t subLevel, bool weightMode, bool allSampleMode)
{
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

    auto points = center.accessor<float, 1>();          
    if (points.size(0) != 3)                            
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(0) << " expect 3" << std::endl;
        return;
    }

    center_[0] = points[0];
    center_[1] = points[1];
    center_[2] = points[2];
    root_ = new Octant;
    root_->x = points[0];
    root_->y = points[1];
    root_->z = points[2];
    root_->extent = extent;    
    root_->depth = 0;           
    extent_ = extent;
    data_ = new std::vector<Eigen::Vector3d>;
    normal_ = new std::vector<Eigen::Vector3d>;
    normalConf_ = new std::vector<bool>;
    curvature_ = new std::vector<double>;
    verts_ = new VertsArray;
    voxel_ = new std::vector<Octant*>;
    mcVoxels_ = new std::unordered_set<Octant*>;
}

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

Octree::Octant::Octant()
    : isLeaf(true), isFixed(0), isFixedLast(0), isUpdated(false), x(0.0f), y(0.0f), z(0.0f), extent(0.0f), depth(0),
    curvature(0.0f), sdf(0), triangles(0), mortonCode(0), size(0), frames(0), weight(0), lastWeight(0), curveWeight(0)
{
    isFixed = new bool[8];      
    isFixedLast = new bool[8];
    frames = new int[8];        
    sdf = new double[8];        
    weight = new uint32_t[8];     
    lastWeight = new uint32_t[8];     
    std::fill(isFixed, isFixed + 8, false);
    std::fill(isFixedLast, isFixedLast + 8, false);
    std::fill(frames, frames + 8, -1);
    std::fill(sdf, sdf + 8, 0);
    std::fill(weight, weight + 8, 0);
    std::fill(lastWeight, lastWeight + 8, 0);
    memset(&child, 0, 8 * sizeof(Octant *));
    memset(&neighbor, 0, 26 * sizeof(Octant *));
}
 
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

void Octree::insert(const Eigen::Matrix<double, Eigen::Dynamic, 3> &points, const Eigen::Vector3d &camera)
{
    std::unordered_set<Octant*> voxels;     
    for (int i = 0; i < points.rows(); ++i)
    {
        insertOctant(points.row(i).transpose(), root_, extent_, 0, voxels);
    }
    normal_->resize(data_->size());     
    normalConf_->resize(data_->size()); 
    curvature_->resize(data_->size());  
    updateNormals(camera, voxels);
    updateVoxel(voxels);
}

void Octree::insertOctant(const Eigen::Vector3d& point, Octant* octant, double extent, uint32_t morton, std::unordered_set<Octant*> &voxels)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    
    if (extent > params_.minExtent)
    {
        if (octant->isLeaf)             
            octant->isLeaf = false;     
        
        uint32_t mortonCode = 0;
        Octant* childOctant;    
        double childExtent;
        if (point[0] > octant->x) mortonCode |= 1;
        if (point[1] > octant->y) mortonCode |= 2;
        if (point[2] > octant->z) mortonCode |= 4;
        if (octant->child[mortonCode])     
        {
            childOctant = octant->child[mortonCode];
            childExtent = childOctant->extent;
        }
        else
        {
            childExtent = 0.5 * extent;      
            static const double factor[] = {-0.5, 0.5};
            double childX = octant->x + factor[(mortonCode & 1) > 0] * extent / 2.0;    
            double childY = octant->y + factor[(mortonCode & 2) > 0] * extent / 2.0;
            double childZ = octant->z + factor[(mortonCode & 4) > 0] * extent / 2.0;
            childOctant = new Octant;
            childOctant->x = childX;              
            childOctant->y = childY;
            childOctant->z = childZ;
            childOctant->extent = childExtent;    
            childOctant->depth = octant->depth + 1;         
            octant->child[mortonCode] = childOctant;
            if (childExtent <= params_.minExtent)       
            {
                voxel_->push_back(childOctant);
                childOctant->mortonCode = (morton <<  3) + mortonCode;
            }
        }
        uint32_t size = childOctant->size;
        morton = (morton <<  3) + mortonCode;     
        insertOctant(point, childOctant, childExtent, morton, voxels);
        if (size != childOctant->size)            
        {
            octant->size += 1;
        }
    }
    else    
    {
        if (octant->size == 0)
        {
            for (uint32_t i = 0; i < 26; ++i)  
            {
                if (octant->neighbor[i] != nullptr)
                    continue;
                auto neighborCode = neighborVoxels(morton, neighborTable[i], 1, octant->depth);
                auto neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);
                octant->neighbor[i] = neighborOctant;
                neighborOctant->neighbor[oppNeighborTableID[i]] = octant;
            }
        }

        for (uint32_t i = 0; i < octant->size; ++i)
        {
            const Eigen::Vector3d& p = points[octant->successors_[i]];      

            auto dis = L2Distance::compute(point, p);                       
            if (dis < std::pow(params_.minSize, 2))                         
                return;

            // auto dis = L1Distance::compute(point, p);
            // if (dis < params_.minSize)                         
            //     return;

        }
        
        octant->size += 1;
        data_->push_back(point); 
        octant->successors_.push_back(data_->size()-1);
        voxels.insert(octant);
        for (uint32_t i = 0; i < 26; ++i)
            voxels.insert(octant->neighbor[i]);
    }
}

Octree::Octant* Octree::createOctant(Octant* octant, uint32_t morton, uint32_t startDepth, const uint32_t maxDepth)
{
    Octant* childOctant;
    double childExtent;

    if (startDepth > maxDepth)
    {
        octant->mortonCode = morton;
        for (uint32_t i = 0; i < 26; ++i)   
        {
            if (octant->neighbor[i] != nullptr)
                continue;
            auto neighborCode = neighborVoxels(morton, neighborTable[i], 1, octant->depth);
            auto neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);
            octant->neighbor[i] = neighborOctant;
            neighborOctant->neighbor[oppNeighborTableID[i]] = octant;
        }
        return octant;      
    }
    
    if (octant->isLeaf)             
        octant->isLeaf = false;     
    auto mortonCode = (morton >> ((maxDepth-startDepth)*3)) & 0x07;
    if (octant->child[mortonCode] == 0)       
    {
        childExtent = 0.5 * octant->extent;      
        static const double factor[] = {-0.5, 0.5};
        double childX = octant->x + factor[(mortonCode & 1) > 0] * octant->extent / 2.0;    
        double childY = octant->y + factor[(mortonCode & 2) > 0] * octant->extent / 2.0;
        double childZ = octant->z + factor[(mortonCode & 4) > 0] * octant->extent / 2.0;
        childOctant = new Octant;
        childOctant->x = childX;              
        childOctant->y = childY;
        childOctant->z = childZ;
        childOctant->extent = childExtent;    
        childOctant->depth = octant->depth + 1;         
        octant->child[mortonCode] = childOctant;
        if (childOctant->depth >= maxDepth)
        {
            voxel_->push_back(childOctant);
        }
    }
    else    
    {
        childOctant = octant->child[mortonCode];
        childExtent = childOctant->extent;
    }
    auto leafOctant = createOctant(childOctant, morton, startDepth+1, maxDepth);

    return leafOctant;
}

Octree::Octant* Octree::createOctantSimply(Octant* octant, uint32_t morton, uint32_t startDepth, const uint32_t maxDepth)
{
    Octant* childOctant;
    double childExtent;

    if (startDepth > maxDepth)
    {
        octant->mortonCode = morton;
        return octant;      
    }
    
    auto mortonCode = (morton >> ((maxDepth-startDepth)*3)) & 0x07;
    if (octant->child[mortonCode] == 0)       
    {
        childExtent = 0.5 * octant->extent;      
        static const double factor[] = {-0.5, 0.5};
        double childX = octant->x + factor[(mortonCode & 1) > 0] * octant->extent / 2.0;    
        double childY = octant->y + factor[(mortonCode & 2) > 0] * octant->extent / 2.0;
        double childZ = octant->z + factor[(mortonCode & 4) > 0] * octant->extent / 2.0;
        childOctant = new Octant;
        childOctant->x = childX;              
        childOctant->y = childY;
        childOctant->z = childZ;
        childOctant->extent = childExtent;    
        childOctant->depth = octant->depth + 1;         
        octant->child[mortonCode] = childOctant;
        if (childOctant->depth >= maxDepth)
        {
            voxel_->push_back(childOctant);
        }
    }
    else    
    {
        childOctant = octant->child[mortonCode];
        childExtent = childOctant->extent;
    }
    auto leafOctant = createOctantSimply(childOctant, morton, startDepth+1, maxDepth);

    return leafOctant;
}

void Octree::updateNormals(const Eigen::Vector3d& camera, std::unordered_set<Octant*> &voxels)
{
    const std::vector<Eigen::Vector3d> &points = *data_;        
    std::vector<Eigen::Vector3d> &normals = *normal_;           
    std::vector<bool> &normalConf = *normalConf_;               
    std::vector<double> &curvature = *curvature_;              
    
    float sqrRadius = L2Distance::sqr(params_.normalRadius);  // "squared" radius
    std::vector<uint32_t> resultIndices;
    
    std::vector<uint32_t> queryIdx;
    for (auto v: voxels) 
    {
        queryIdx.insert(queryIdx.end(), v->successors_.begin(), v->successors_.end());
    }
    
    #pragma omp parallel for private(resultIndices)
    for (uint32_t i = 0; i < queryIdx.size(); ++i) 
    {
        if (normalConf[queryIdx[i]])      
            continue;
        resultIndices.clear();
        radiusNeighbors(root_, points[queryIdx[i]], params_.normalRadius, sqrRadius, resultIndices);
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
        
        normals[queryIdx[i]] = normal;
        normalConf[queryIdx[i]] = true;
        curvature[queryIdx[i]] = curve;
    }
}

void Octree::updateCurvature(Octant* octant)
{
    const std::vector<Eigen::Vector3d>& normals = *normal_;
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
        octant->curvature = eigenvalue[0]/eigenvalue[2];        
}


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

    // Completely outside, since q' is outside the relevant area.   
    if (x > maxdist || y > maxdist || z > maxdist) return false;

    int32_t num_less_extent = (x < (o->extent)/2.0f) + (y < (o->extent)/2.0f) + (z < (o->extent)/2.0f);    

    // Checking different cases:

    // a. inside the surface region of the octant. 
    if (num_less_extent > 1) return true;

    // b. checking the corner region && edge region.
    x = std::max(x - (o->extent)/2.0f, 0.0);
    y = std::max(y - (o->extent)/2.0f, 0.0);
    z = std::max(z - (o->extent)/2.0f, 0.0);

    return (L2Distance::norm(x, y, z) < sqRadius);
}

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

void Octree::radiusNeighbors(const Octant* octant, const Eigen::Vector3d& query, double radius,
                                                 double sqrRadius, std::vector<uint32_t>& resultIndices) const
{
    const std::vector<Eigen::Vector3d>& points = *data_;

    if (contains(query, sqrRadius, octant))             
    {
        std::vector<uint32_t> pointVec;
        subSuccessors(octant, pointVec);
        for (uint32_t i = 0; i < pointVec.size(); ++i)     
        {
            resultIndices.push_back(pointVec[i]);          
        }
        return;  
    }

    if (octant->isLeaf)     
    {
        for (uint32_t i = 0; i < octant->size; ++i)
        {
            uint32_t pointIdx = octant->successors_[i];     
            const Eigen::Vector3d& p = points[pointIdx];    
            double dist = L2Distance::compute(query, p);     
            if (dist < sqrRadius)       
            {
                resultIndices.push_back(pointIdx);
            }
        }
        return;     
    }

    for (uint32_t c = 0; c < 8; ++c)
    {
        if (octant->child[c] == 0) continue;
        if (!overlaps(query, radius, sqrRadius, octant->child[c])) continue;
        radiusNeighbors(octant->child[c], query, radius, sqrRadius, resultIndices);
    }
}

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
            uint32_t pointIdx = octant->successors_[i];     
            if (normalConf[pointIdx])
            {
                const Eigen::Vector3d& p = points[pointIdx];    
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


void Octree::trianglesClear(Octant* octant)
{
    VertsArray& verts = *verts_;
    for (std::vector<Eigen::Vector3i>::iterator it = octant->triangles->begin(); it != octant->triangles->end(); ++it)
    {
        Eigen::Vector3i& triangles = *it;
        for (uint32_t k = 0; k < 3 ; ++k)   
        {
            verts.remove(triangles[k], octant->mortonCode);
        }
    }
    octant->triangles->clear();
}

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

void Octree::marchingCubesSparse(Octant* octant, const Eigen::Vector3i* valid_cords, const Eigen::Tensor<double, 3> &dense_sdf,
                                 const std::vector<bool> mask, const uint32_t level,const Eigen::Vector3d offsets)
{
    if (octant->triangles == nullptr)
        octant->triangles = new std::vector<Eigen::Vector3i>;       
    else
    {
        if (!params_.meshOverlap)               
            trianglesClear(octant);
    }
        
    VertsArray& verts = *verts_;
    std::vector<Eigen::Vector3i> neighborTriangles;
    neighborTriangles.insert(neighborTriangles.end(), octant->triangles->begin(), octant->triangles->end());
    for (uint32_t i = 0; i < 26; ++i)
    {
        auto neighborOctant = octant->neighbor[i];
        if (neighborOctant == 0 || neighborOctant->triangles == 0 || neighborOctant->triangles->size() == 0)
            continue;
        neighborTriangles.insert(neighborTriangles.end(), neighborOctant->triangles->begin(), neighborOctant->triangles->end());
    }
    
    std::vector<Eigen::Vector3d> vp;       
    marchingCubesDense(vp, valid_cords, dense_sdf, mask, level*level*level, offsets, octant->extent/level);
    for (uint32_t i = 0; i < vp.size()/3; ++i)
    {
        Eigen::Vector3i vt;         
        for (uint32_t j = 0; j < 3; ++j)
        {
            bool isRepeat = false;
            uint32_t repeatNum;
            for (uint32_t it = 0; it < neighborTriangles.size(); ++it)
            {
                Eigen::Vector3i triangles = neighborTriangles[it];   
                for (uint32_t k = 0; k < 3 ; ++k)   
                {
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
        neighborTriangles.push_back(vt);      
    }
}

void Octree::updateVoxel(std::unordered_set<Octant*> &voxels)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    const std::vector<bool> &normalConf = *normalConf_;
    const std::vector<double> &curvature = *curvature_;

    std::vector<Octant*> voxelsList;
    voxelsList.insert(voxelsList.end(), voxels.begin(), voxels.end());     

    #pragma omp parallel for
    for (uint32_t i = 0; i < voxelsList.size(); ++i)
    {
        Octant* childOctant = voxelsList[i];
        double cubeSDF[8];                             
        double cubeValid[8];                           

        double extent = extent_;         
        while(extent > params_.minExtent)
        {
            extent /= 2.0;
        }

        if (childOctant->size > 0)
        {
            double sumCurvature = 0;
            uint32_t count = 0;
            for (const auto pointIdx : childOctant->successors_)
            {
                if (normalConf[pointIdx])       
                {
                    sumCurvature += curvature[pointIdx];
                    count++;
                }
            }
            childOctant->curvature = sumCurvature / (double)count;
        }

        for (int a = 0; a < 2; ++a)         
        {
            double x = childOctant->x + (a % 2 == 0 ? -1 : 1) * extent / 2.0;
            for (int b = 0; b < 2; ++b)     
            {
                double y = childOctant->y + (b % 2 == 0 ? -1 : 1) * extent / 2.0;
                for (int c = 0; c < 2; ++c) 
                {
                    double z = childOctant->z + (c % 2 == 0 ? -1 : 1) * extent / 2.0;

                    double distances = INFINITY;
                    uint32_t nearIdx = 0;
                    findNeighbor(root_, Eigen::Vector3d(x, y, z), -1, distances, nearIdx);
                    
                    auto nearPoint = points[nearIdx];   
                    auto nearNormal = normals[nearIdx]; 
                    double distance = (Eigen::Vector3d(x, y, z) - nearPoint).dot(nearNormal);
                    if (std::abs(distances) < params_.sdfRadius)
                    {
                        cubeSDF[4*a+2*b+c] = distance;
                        cubeValid[4*a+2*b+c] = 1;
                    }
                    else
                    {
                        cubeValid[4*a+2*b+c] = 0;             
                    }
                }
            }
        }
        std::copy(cubeSDF, cubeSDF + 8, childOctant->sdf);
        std::copy(cubeValid, cubeValid + 8, childOctant->weight);       
    }
}

void Octree::fusionPoints(Octree &tree, Octant *baseOctant, Octant *treeOctant)
{
    const std::vector<Eigen::Vector3d> &treePoints = *(tree.data_);        
    const std::vector<Eigen::Vector3d> &treeNormals = *(tree.normal_);           
    const std::vector<bool> &treeNormalConf = *(tree.normalConf_);           
    const std::vector<Eigen::Vector3d> &basePoints = *data_;       

    for (uint32_t i = 0; i < treeOctant->size; ++i) 
    {
        auto point = treePoints[treeOctant->successors_[i]];
        auto normal = treeNormals[treeOctant->successors_[i]];
        auto normalConf = treeNormalConf[treeOctant->successors_[i]];
        // 遍历该voxel中的所有点 判断是否满足插入距离
        bool isValid = true;
        for (uint32_t j = 0; j < baseOctant->size; ++j)
        {
            const Eigen::Vector3d& p = basePoints[baseOctant->successors_[j]];      

            auto dis = L2Distance::compute(point, p);                      
            if (dis < std::pow(params_.minSize, 2))                         
            {
                isValid = false;
                break;
            }
            // L1
            // auto dis = L1Distance::compute(point, p);
            // if (dis < params_.minSize)                        
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

    if (baseOctant->curveWeight == 0)       
    {
        baseOctant->curvature = treeOctant->curvature;
        baseOctant->curveWeight = treeOctant->size;
    }
    else                                    
    {
        baseOctant->curvature = ((baseOctant->curveWeight * baseOctant->curvature) + (treeOctant->size * treeOctant->curvature)) \
                                / (baseOctant->curveWeight + treeOctant->size);
        baseOctant->curveWeight += treeOctant->size;
    }
}


bool Octree::maskEdgeVoxel(Octant* octant, std::vector<bool> &subMask, int level)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const VertsArray& verts = *verts_;
    std::unordered_set<uint32_t> faceIds;
    bool isDetial = true;

    for (uint32_t i = 0; i < octant->triangles->size(); ++i)
    {
        faceIds.insert((*(octant->triangles))[i][0]);
        faceIds.insert((*(octant->triangles))[i][1]);
        faceIds.insert((*(octant->triangles))[i][2]);
    }
    
    for (auto fid : faceIds)
    {
        isDetial = true;
        auto vt = verts[fid].point;
        for (uint32_t i = 0; i < octant->size; ++i)        
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
                for (uint32_t j = 0; j < neighborOctant->size; ++j)        
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
    if (isDetial)       
    {
        Eigen::Vector3d offsets(octant->x-(octant->extent/2), octant->y-(octant->extent/2), octant->z-(octant->extent/2));
        double scale = octant->extent / level;
        for (uint32_t i = 0; i < octant->size; ++i)
        {
            auto p = points[octant->successors_[i]];
            auto op = (p - offsets) / scale;
            int cellX = static_cast<int>(op[0]);
            int cellY = static_cast<int>(op[1]);
            int cellZ = static_cast<int>(op[2]);
            if (cellX >= 0 && cellX < level && cellY >= 0 && cellY < level && cellZ >= 0 && cellZ < level)      
            {
                subMask[cellX * level * level + cellY * level + cellZ] = false;
            }
        }
        return true;
    }
    return false;
}

void Octree::SDFdeviation(const Octant* octant, Eigen::Vector3i *subIndex, Eigen::Tensor<double, 3> &subSDF, int level)
{
    const std::vector<Eigen::Vector3d>& points = *data_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    subSDF(0, 0, 0) = octant->sdf[0];
    subSDF(0, 0, level) = octant->sdf[1];
    subSDF(0, level, 0) = octant->sdf[2];
    subSDF(0, level, level) = octant->sdf[3];
    subSDF(level, 0, 0) = octant->sdf[4];
    subSDF(level, 0, level) = octant->sdf[5];
    subSDF(level, level, 0) = octant->sdf[6];
    subSDF(level, level, level) = octant->sdf[7];
    if (params_.curvatureTHR == -1 ||octant->curvature < params_.curvatureTHR)      
    {
        for (int a = 0; a <= level; ++a) {
            for (int b = 0; b <= level; ++b) {
                for (int c = 0; c <= level; ++c) {
                    if (a != level && b != level && c != level)
                        subIndex[a * level * level + b * level + c] = Eigen::Vector3i(a, b, c); 

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
    else        
    {
        for (int a = 0; a <= level; ++a) {
            for (int b = 0; b <= level; ++b) {
                for (int c = 0; c <= level; ++c) {
                    if (a != level && b != level && c != level)
                        subIndex[a * level * level + b * level + c] = Eigen::Vector3i(a, b, c); 
                    bool reCal = true;          
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
                        || (a == level  && b == 0 && c == 0) || (a == level  && b == 0 && c == level) || (a == level  && b == level && c == 0) || (a == level  && b == level && c == level))       
                        continue;
                    if (reCal)
                    {
                        auto orign = Eigen::Vector3d(octant->x - (octant->extent / 2), octant->y - (octant->extent / 2), octant->z - (octant->extent / 2));
                        double subExtent = octant->extent / level;
                        auto query = Eigen::Vector3d(orign[0] + a*subExtent, orign[1] + b*subExtent, orign[2] + c*subExtent);
                        double distances = INFINITY;
                        uint32_t nearIdx = 0;
                        findNeighbor(root_, query, -1, distances, nearIdx);
                        auto nearPoint = points[nearIdx];   
                        auto nearNormal = normals[nearIdx]; 
                        subSDF(a, b, c) = (query - nearPoint).dot(nearNormal);
                    }
                    else    
                    {
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

void Octree::voxelStatusCheck(Octree &tree)
{
    const std::vector<Octant*>& treeVoxels = *(tree.voxel_);
    for (auto treeOctant : treeVoxels)
    {
        if (treeOctant->size == 0)          
            continue;
        auto code = treeOctant->mortonCode; 
        auto baseOctant = createOctant(root_, code, 1, treeOctant->depth);      
        bool isFixed = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (!baseOctant->isFixedLast[i])
            {
                isFixed = false;
                break;
            }
        }
        if (isFixed)    
            continue;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (baseOctant->isFixedLast[i])         
            {
                baseOctant->isFixed[i] = false;
                for (uint32_t j = 0; j < 7; ++j)    
                {
                    auto neighborOctant = baseOctant->neighbor[cubeVertsNeighborTable[i][j]];
                    if (neighborOctant)
                        neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;
                }
            }
        }
    }  
}

void Octree::fusionVoxel(Octree &tree, int frames)
{
    const std::vector<Octant*>& treeVoxels = *(tree.voxel_);

    if (!params_.weightMode)        
        voxelStatusCheck(tree);

    for (auto treeOctant : treeVoxels)
    {
        auto code = treeOctant->mortonCode;
        auto octant = createOctant(root_, code, 1, treeOctant->depth);
        if (params_.allSampleMode)
        {
            mcVoxels_->insert(octant);
        }

        if (treeOctant->size != 0)              
        {
            fusionPoints(tree, octant, treeOctant);     

            for (uint32_t i = 0; i < 8; ++i)    
            {
                if (treeOctant->weight[i] == 0)    
                    continue;
                if (octant->frames[i] == -1)        
                {   
                    octant->frames[i] = frames;
                    long weight = treeOctant->size;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            weight += neighborOctant->size;
                    }
                    octant->weight[i] = weight;                 
                    octant->sdf[i] = treeOctant->sdf[i];        
                    octant->lastWeight[i] = weight;             
                    for (uint32_t j = 0; j < 7; ++j)        
                    {
                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (!neighborOctant)                
                        {
                            auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                            neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    
                        }
                        neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                        neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  
                        neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        
                        neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        
                    }
                }
                else if (octant->frames[i] != frames)   
                {
                    octant->frames[i] = frames;
                    long weight = treeOctant->size;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            weight += neighborOctant->size;
                    }
                    if (octant->isFixed[i])
                    {
                        if (params_.weightMode)     
                        {
                            if (weight - octant->lastWeight[i] < params_.reconTHR)
                            {
                                continue;
                            }
                            else
                            {
                                octant->isFixed[i] = false;
                                octant->isUpdated = false;
                                for (uint32_t j = 0; j < 7; ++j)        
                                {
                                    auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                                    neighborOctant->isUpdated = false;
                                    neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;
                                }
                            }
                        }
                        else                        
                        {
                            continue;
                        }
                    }
                    if (!params_.weightMode)        
                    {
                        octant->isUpdated = false;
                        for (uint32_t j = 0; j < 7; ++j)        
                        {
                            auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                            neighborOctant->isUpdated = false;
                            neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;	 
                        }
                    }
                    double sdf = ((octant->weight[i] * octant->sdf[i]) + (weight * treeOctant->sdf[i])) / (octant->weight[i] + weight);
                    octant->weight[i] = octant->weight[i] + weight;     
                    octant->sdf[i] = sdf;                               
                    octant->lastWeight[i] = weight;                     

                    for (uint32_t j = 0; j < 7; ++j)        
                    {
                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (!neighborOctant)                
                        {
                            auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                            neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    
                        }
                        neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                        neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  
                        neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        
                        neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        
                    }
                }
            }
        }
        else
        {
            for (uint32_t i = 0; i < 8; ++i)    
            {
                if (treeOctant->weight[i] == 0)    
                    continue;
                
                if (octant->frames[i] == -1)        
                {
                    octant->frames[i] = frames;
                    long weight = treeOctant->size;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            weight += neighborOctant->size;
                    }
                    octant->weight[i] = weight;                 
                    octant->sdf[i] = treeOctant->sdf[i];        
                    octant->lastWeight[i] = weight;             
                    for (uint32_t j = 0; j < 7; ++j)        
                    {
                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (!neighborOctant)                
                        {
                            auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                            neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);    
                        }
                        neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                        neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  
                        neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        
                        neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        
                    }
                }
                else if (octant->frames[i] != frames)   
                {
                    bool hasCenter = false;
                    for (uint32_t j = 0; j < 7; ++j)
                    {
                        auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                        if (neighborOctant && neighborOctant->size > 0)
                            hasCenter = true;
                    }
                    if (hasCenter)          
                    {
                        octant->frames[i] = frames;
                        long weight = treeOctant->size;
                        for (uint32_t j = 0; j < 7; ++j)
                        {
                            auto neighborOctant = treeOctant->neighbor[cubeVertsNeighborTable[i][j]];
                            if (neighborOctant && neighborOctant->size > 0)
                                weight += neighborOctant->size;
                        }
                        if (octant->isFixed[i])
                        {
                            if (params_.weightMode)      
                            {
                                if (weight - octant->lastWeight[i] < params_.reconTHR)
                                {
                                    continue;
                                }
                                else
                                {
                                    octant->isFixed[i] = false;
                                    octant->isUpdated = false;
                                    for (uint32_t j = 0; j < 7; ++j)        
                                    {
                                        auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                                        neighborOctant->isUpdated = false;
                                        neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false;
                                    }
                                }
                            }
                            else                        
                            {
                                continue;
                            }
                        }
                        if (!params_.weightMode)        
                        {
                            octant->isUpdated = false;
                            for (uint32_t j = 0; j < 7; ++j)        
                            {
                                auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                                neighborOctant->isUpdated = false;
                                neighborOctant->isFixed[cubeVertsNeighborVTable[i][j]] = false; 
                            }
                        }
                        double sdf = ((octant->weight[i] * octant->sdf[i]) + (weight * treeOctant->sdf[i])) / (octant->weight[i] + weight);
                        octant->weight[i] = octant->weight[i] + weight;     
                        octant->sdf[i] = sdf;                               
                        octant->lastWeight[i] = weight;                     
                        for (uint32_t j = 0; j < 7; ++j)        
                        {
                            auto neighborOctant = octant->neighbor[cubeVertsNeighborTable[i][j]];
                            if (!neighborOctant)                
                            {
                                auto neighborCode = neighborVoxels(octant->mortonCode, neighborTable[cubeVertsNeighborTable[i][j]], 1, octant->depth);
                                neighborOctant = createOctantSimply(root_, neighborCode, 1, octant->depth);   
                            }
                            neighborOctant->frames[cubeVertsNeighborVTable[i][j]] = frames;
                            neighborOctant->weight[cubeVertsNeighborVTable[i][j]] = octant->weight[i];  
                            neighborOctant->sdf[cubeVertsNeighborVTable[i][j]] = octant->sdf[i];        
                            neighborOctant->lastWeight[cubeVertsNeighborVTable[i][j]] = octant->lastWeight[i];        
                        }
                    }
                }
            }
        }
    }
    if ((frames + 1) % params_.mcInterval)
        return;
    const std::vector<Octant*>& voxels = *voxel_;         
    #pragma omp parallel for
    for (auto octant : voxels)
    {
        bool isValid = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (octant->weight[i] == 0)
            {
                isValid = false;
                break;
            }
        }
        if (!isValid || octant->isUpdated)      
            continue;
        int level = params_.subLevel;      
        Eigen::Tensor<double, 3> subSDF(level+1, level+1 , level+1);     
        Eigen::Vector3i subIndex[level * level * level];                
        std::vector<bool> subMask(level * level * level, false);      
        Eigen::Vector3d offsets(octant->x-(octant->extent/2), octant->y-(octant->extent/2), octant->z-(octant->extent/2));
        if (level > 1)                          
        {
            SDFdeviation(octant, subIndex, subSDF, level);         
        }
        else                                    
        {
            int num = 0;
            for (int a = 0; a <= 1; ++a) 
            {
                for (int b = 0; b <= 1; ++b) 
                {
                    for (int c = 0; c <= 1; ++c) 
                    {
                        if (a != 1 && b != 1 && c != 1)
                            subIndex[a * 1 * 1 + b * 1 + c] = Eigen::Vector3i(a, b, c); 
                        subSDF(a, b, c) = octant->sdf[num++];
                    }
                }
            }
        }
        #pragma omp critical
        {
            marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);        
        }
        if (level > 1)
        {
            subMask.assign(level * level * level, true);        
            if (maskEdgeVoxel(octant, subMask, level))          
            {
                #pragma omp critical
                {
                    trianglesClear(octant);                     
                    marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);    
                }
            }
        }
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
    std::vector<Eigen::Vector3i> triangles;     

    const std::vector<Octant*>& voxels = *voxel_;        
    for (uint32_t i = 0; i < voxels.size(); ++i)          
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Octree::packOutput()
{
    const VertsArray& verts = *verts_;
    const std::unordered_set<Octant*>& voxels = *mcVoxels_;
    const std::vector<Eigen::Vector3d>& normals = *normal_;
    const std::vector<bool>& normalConf = *normalConf_;               
    const std::vector<Eigen::Vector3d>& points = *data_;
    std::unordered_map<int, int> faceMap;                   
    std::unordered_set<Octant*> fliterVoxels;             
    torch::Tensor trianglesTensor = torch::zeros({static_cast<int64_t>(voxels.size()*24), 3}, torch::kInt);     
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

    for (auto octant : voxels)
    {
        if (octant->triangles == nullptr)
            continue;
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
        for (uint32_t i = 0; i<(*(octant->triangles)).size(); ++i)      
        {
            for (uint32_t j = 0; j < 3; ++j)
            {
                if (faceMap.find((*(octant->triangles))[i][j]) == faceMap.end())     
                {
                    trianglesTensor[faceNum][j] = static_cast<int>(faceMap.size());
                    faceMap[(*(octant->triangles))[i][j]] = static_cast<int>(faceMap.size());
                    indexTensor[vertsNum] = ((*(octant->triangles))[i][j]);
                    vertsTensor[vertsNum][0] = verts[((*(octant->triangles))[i][j])].point[0];
                    vertsTensor[vertsNum][1] = verts[((*(octant->triangles))[i][j])].point[1];
                    vertsTensor[vertsNum][2] = verts[((*(octant->triangles))[i][j])].point[2];
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
            if (isMask)
            {
                trianglesMaskTensor[faceNum] = true;
            }
            faceNum++;
        }
        if (isMask && (*(octant->triangles)).size() > 0)
        {
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
    //             if (faceMap.find((*(octant->triangles))[i][j]) == faceMap.end())      
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
    //         trianglesMaskTensor[faceNum] = true;
    //         faceNum++;
    //     }
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

    // for (auto octant : voxels)
    // {
    //     octant->processFlags = false;
    // }

    mcVoxels_->clear();

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
    std::vector<bool>& normalConf = *normalConf_;               

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Octree::packCubeSDF()
{
    int64_t depth = 0;                  
    double extent = extent_;
    while(extent > params_.minExtent)
    {
        extent /= 2.0;
        depth++;
    }
    int64_t level = params_.subLevel;           
    int64_t csize = static_cast<int64_t>(std::pow(2, depth));
    torch::Tensor sdfCube = torch::zeros({(csize * level) + 1, (csize * level) + 1, (csize * level) + 1}, torch::kDouble);
    torch::Tensor maskCube = torch::zeros({(csize * level) + 1, (csize * level) + 1, (csize * level) + 1}, torch::kBool);
    torch::Tensor offsets = torch::zeros({4}, torch::kDouble);
    offsets[0] = center_[0] - (extent_ / 2);
    offsets[1] = center_[1] - (extent_ / 2);
    offsets[2] = center_[2] - (extent_ / 2);
    offsets[3] = extent_ / (csize * level);
    const std::vector<Octant*>& voxels = *voxel_;
    #pragma omp parallel for
    for (auto octant : voxels)
    {
        bool isValid = true;
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (octant->weight[i] == 0)
            {
                isValid = false;
                break;
            }
        }
        if (!isValid || octant->isUpdated)      
            continue;
        Eigen::Tensor<double, 3> subSDF(level+1, level+1 , level+1);     
        Eigen::Vector3i subIndex[level * level * level];                
        std::vector<bool> subMask(level * level * level, false);      
        Eigen::Vector3d offsets(octant->x-(octant->extent/2), octant->y-(octant->extent/2), octant->z-(octant->extent/2));
        if (level > 1)                         
        {
            SDFdeviation(octant, subIndex, subSDF, level);         
        }
        else                                    
        {
            int num = 0;
            for (int a = 0; a <= 1; ++a) 
            {
                for (int b = 0; b <= 1; ++b) 
                {
                    for (int c = 0; c <= 1; ++c) 
                    {
                        if (a != 1 && b != 1 && c != 1)
                            subIndex[a * 1 * 1 + b * 1 + c] = Eigen::Vector3i(a, b, c); 
                        subSDF(a, b, c) = octant->sdf[num++];
                    }
                }
            }
        }
        #pragma omp critical
        {
            marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);         
        }
        if (level > 1)
        {
            subMask.assign(level * level * level, true);        
            if (maskEdgeVoxel(octant, subMask, level))          
            {
                #pragma omp critical
                {
                    trianglesClear(octant);                     
                    marchingCubesSparse(octant, subIndex, subSDF, subMask, level, offsets);     
                }
            }
            else                                                
            {
                subMask.assign(level * level * level, false);
            }
        }
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
        auto code = octant->mortonCode;
        double extent = csize/2;
        Eigen::Vector3d index(csize/2, csize/2, csize/2);      
        for (int i = depth-1; i >= 0; --i)
        {
            auto tmpCode = (code >> (i*3)) & 0x07;   
            int bit2 = (tmpCode >> 2) & 1; 
            int bit1 = (tmpCode >> 1) & 1; 
            int bit0 = tmpCode & 1;       
            extent /= 2;
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
        index[0] -= 0.5; index[1] -= 0.5; index[2] -= 0.5;   
        index[0] *= level; index[1] *= level; index[2] *= level;

        for (int a = 0; a <= level; ++a) {
            for (int b = 0; b <= level; ++b) {
                for (int c = 0; c <= level; ++c) {
                    sdfCube[index[0] + a][index[1] + b][index[2] + c] = subSDF(a, b, c);        
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
    torch::Tensor voxelTensor = torch::zeros({static_cast<int64_t>(voxels.size()), 8, 3}, torch::kDouble);  
    torch::Tensor sdfTensor = torch::zeros({static_cast<int64_t>(voxels.size()), 8}, torch::kDouble);       
    torch::Tensor codeTensor = torch::zeros({static_cast<int64_t>(voxels.size())}, torch::kLong);         
    uint32_t i = 0;
    for (auto octant : voxels)    
    {
        if (octant->sdf == nullptr)
            continue;

        double extent = extent_;         
        while(extent > params_.minExtent)
        {
            extent /= 2.0;
        }

        for (int a = 0; a < 2; ++a)         // x
        {
            double x = octant->x + (a % 2 == 0 ? -1 : 1) * extent / 2.0;
            for (int b = 0; b < 2; ++b)     // y
            {
                double y = octant->y + (b % 2 == 0 ? -1 : 1) * extent / 2.0;
                for (int c = 0; c < 2; ++c) // z
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

void Octree::updateTree(const torch::Tensor depth, const torch::Tensor K, const torch::Tensor Rt, int64_t frames)
{
    auto pointsT = transformToPointCloud(depth, K, Rt);
    auto points = libtorch2eigen<float>(pointsT);
    // auto T = Rt.slice(1, 3, 4).slice(0, 0, 3).squeeze(1);   
    auto cameraRt = Rt.accessor<float, 2>();
    Eigen::Vector3d camera(cameraRt[0][3], cameraRt[1][3], cameraRt[2][3]);
    Octree frameTree(center_, extent_, params_);     
    frameTree.insert(points.cast<double>(), camera);    
    fusionVoxel(frameTree, frames);
}

void Octree::updateTreePcd(const torch::Tensor pointData, const torch::Tensor K, const torch::Tensor Rt, int64_t frames)
{
    auto camera_coords = torch::cat({pointData, torch::ones({pointData.size(0), 1})}, 1);
    auto camera_coordsT = camera_coords.transpose(1, 0);
    torch::Tensor world_coords = Rt.mm(camera_coordsT);
    torch::Tensor pointWorld = world_coords.t().slice(1, 0, 3);
    auto points = libtorch2eigen<float>(pointWorld);  
    auto cameraRt = Rt.accessor<float, 2>();
    Eigen::Vector3d camera(cameraRt[0][3], cameraRt[1][3], cameraRt[2][3]);
    Octree frameTree(center_, extent_, params_);     
    frameTree.insert(points.cast<double>(), camera);    
    fusionVoxel(frameTree, frames);
}

